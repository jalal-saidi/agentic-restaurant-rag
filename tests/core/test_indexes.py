from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from connoisseur.core.indexes import ChromaIndex, SearchDocument


class FakeCollection:
    def __init__(self) -> None:
        self.ids = {"stale"}
        self.deleted: list[str] = []
        self.metadata_by_id: dict[str, dict] = {}
        self.upsert_calls = 0

    def upsert(self, *, ids, documents, metadatas) -> None:
        self.upsert_calls += 1
        self.ids.update(ids)
        self.documents = documents
        self.metadatas = metadatas
        self.metadata_by_id.update(dict(zip(ids, metadatas, strict=True)))

    def get(self, *, include=None) -> dict:
        ids = sorted(self.ids)
        return {
            "ids": ids,
            "metadatas": [self.metadata_by_id.get(item) for item in ids],
        }

    def delete(self, *, ids) -> None:
        self.deleted.extend(ids)
        self.ids.difference_update(ids)

    def query(self, **kwargs) -> dict:
        self.query_arguments = kwargs
        return {
            "ids": [["restaurant:2", "restaurant:1"]],
            "distances": [[0.1, 0.5]],
        }


class FakeChromaClient:
    def __init__(self) -> None:
        self.collection = FakeCollection()
        self.collection_arguments: dict | None = None

    def get_or_create_collection(self, **kwargs) -> FakeCollection:
        self.collection_arguments = kwargs
        return self.collection


class ChromaIndexTests(unittest.TestCase):
    def test_fake_client_exercises_persistence_and_filtered_ranking(self) -> None:
        documents = (
            SearchDocument("restaurant:1", "garden table", {"kind": "restaurant"}),
            SearchDocument("restaurant:2", "noodle bar", {"kind": "restaurant"}),
        )
        client = FakeChromaClient()
        with tempfile.TemporaryDirectory() as directory:
            index = ChromaIndex(
                documents,
                persist_path=Path(directory),
                collection_name="test_restaurants",
                client=client,
            )
            hits = index.search(
                "noodles",
                limit=2,
                allowed_ids={"restaurant:1"},
            )

        self.assertEqual(client.collection.deleted, ["stale"])
        self.assertEqual(hits, [("restaurant:1", 1 / 1.5)])
        self.assertEqual(
            client.collection_arguments["configuration"],
            {"hnsw": {"space": "cosine"}},
        )

    def test_unchanged_persistent_documents_are_not_reembedded(self) -> None:
        documents = (
            SearchDocument("recipe:1", "tomato soup", {"kind": "recipe"}),
        )
        client = FakeChromaClient()
        with tempfile.TemporaryDirectory() as directory:
            arguments = {
                "persist_path": Path(directory),
                "collection_name": "test_recipes",
                "client": client,
            }
            ChromaIndex(documents, **arguments)
            ChromaIndex(documents, **arguments)

        self.assertEqual(client.collection.upsert_calls, 1)


if __name__ == "__main__":
    unittest.main()
