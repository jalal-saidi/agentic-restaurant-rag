"""Dependency-light lexical and optional Chroma vector indexes."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_CONTENT_HASH_KEY = "connoisseur_content_hash"


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(text.casefold()))


@dataclass(frozen=True, slots=True)
class SearchDocument:
    id: str
    text: str
    metadata: Mapping[str, str | int | float | bool] = field(default_factory=dict)


class SearchIndex(Protocol):
    backend_name: str

    def search(
        self,
        query: str,
        *,
        limit: int,
        allowed_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        ...


class LexicalIndex:
    """Small in-memory TF-IDF index used locally and as a safe fallback."""

    backend_name = "lexical"

    def __init__(self, documents: Iterable[SearchDocument]) -> None:
        self._documents = tuple(documents)
        self._by_id = {document.id: document for document in self._documents}
        if len(self._by_id) != len(self._documents):
            raise ValueError("Search document IDs must be unique")
        self._tokens = {
            document.id: Counter(tokenize(document.text))
            for document in self._documents
        }
        self._document_frequency: Counter[str] = Counter()
        for counts in self._tokens.values():
            self._document_frequency.update(counts.keys())

    def _idf(self, token: str) -> float:
        return math.log(
            (len(self._documents) + 1)
            / (self._document_frequency.get(token, 0) + 1)
        ) + 1.0

    def search(
        self,
        query: str,
        *,
        limit: int,
        allowed_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        candidates = (
            self._documents
            if allowed_ids is None
            else tuple(
                document
                for document in self._documents
                if document.id in allowed_ids
            )
        )
        if not candidates or limit <= 0:
            return []

        query_tokens = Counter(tokenize(query))
        if not query_tokens:
            return [(document.id, 0.0) for document in candidates[:limit]]

        query_weights = {
            token: (1.0 + math.log(count)) * self._idf(token)
            for token, count in query_tokens.items()
        }
        query_norm = math.sqrt(sum(weight**2 for weight in query_weights.values()))
        normalized_query = " ".join(tokenize(query))
        scored: list[tuple[str, float]] = []
        for document in candidates:
            counts = self._tokens[document.id]
            document_weights = {
                token: (1.0 + math.log(counts[token])) * self._idf(token)
                for token in query_weights
                if counts.get(token)
            }
            dot_product = sum(
                query_weights[token] * weight
                for token, weight in document_weights.items()
            )
            document_norm = math.sqrt(
                sum(
                    ((1.0 + math.log(count)) * self._idf(token)) ** 2
                    for token, count in counts.items()
                )
            )
            score = (
                dot_product / (query_norm * document_norm)
                if dot_product and document_norm and query_norm
                else 0.0
            )
            document_text = " ".join(tokenize(document.text))
            if normalized_query and normalized_query in document_text:
                score += 0.25
            if score > 0:
                scored.append((document.id, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        if scored:
            return scored[:limit]
        # A lexical fallback should still produce deterministic candidates for
        # natural-language queries whose wording is absent from the corpus.
        return [(document.id, 0.0) for document in candidates[:limit]]


class ChromaIndex:
    """Persistent Chroma index using Chroma's default local embeddings."""

    backend_name = "semantic"

    def __init__(
        self,
        documents: Iterable[SearchDocument],
        *,
        persist_path: Path,
        collection_name: str,
        client: Any = None,
        embedding_function: Callable[..., Any] | None = None,
    ) -> None:
        self._documents = tuple(documents)
        self._by_id = {document.id: document for document in self._documents}
        if len(self._by_id) != len(self._documents):
            raise ValueError("Search document IDs must be unique")

        if client is None:
            chromadb = importlib.import_module("chromadb")
            persist_path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(persist_path))
        collection_kwargs: dict[str, Any] = {
            "name": collection_name,
            "configuration": {"hnsw": {"space": "cosine"}},
        }
        if embedding_function is not None:
            collection_kwargs["embedding_function"] = embedding_function
        self._collection = client.get_or_create_collection(**collection_kwargs)
        self._synchronize()
        self._verify_query_capability()

    def _verify_query_capability(self) -> None:
        """Fail initialization if the query embedding model is unavailable."""

        if not self._documents:
            return
        self._collection.query(
            query_texts=["connoisseur readiness"],
            n_results=1,
            include=["distances"],
        )

    def _synchronize(self) -> None:
        stored_result = self._collection.get(include=["metadatas"])
        stored_ids = stored_result.get("ids", []) or []
        stored_metadatas = stored_result.get("metadatas", []) or []
        stored_hashes = {
            document_id: (
                stored_metadatas[index].get(_CONTENT_HASH_KEY)
                if index < len(stored_metadatas) and stored_metadatas[index]
                else None
            )
            for index, document_id in enumerate(stored_ids)
        }

        document_metadata: dict[str, dict[str, Any]] = {}
        changed_documents: list[SearchDocument] = []
        for document in self._documents:
            canonical = json.dumps(
                {
                    "text": document.text,
                    "metadata": dict(document.metadata),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            metadata = {
                **document.metadata,
                _CONTENT_HASH_KEY: content_hash,
            }
            document_metadata[document.id] = metadata
            if stored_hashes.get(document.id) != content_hash:
                changed_documents.append(document)

        if changed_documents:
            self._collection.upsert(
                ids=[document.id for document in changed_documents],
                documents=[document.text for document in changed_documents],
                metadatas=[
                    document_metadata[document.id]
                    for document in changed_documents
                ],
            )
        current_ids = {document.id for document in self._documents}
        stale = sorted(set(stored_ids).difference(current_ids))
        if stale:
            self._collection.delete(ids=stale)

    def search(
        self,
        query: str,
        *,
        limit: int,
        allowed_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        if limit <= 0 or not self._documents:
            return []
        if not tokenize(query):
            candidate_ids = [
                document.id
                for document in self._documents
                if allowed_ids is None or document.id in allowed_ids
            ]
            return [(document_id, 0.0) for document_id in candidate_ids[:limit]]

        # The bundled corpora are small. Fetching the whole ranked set keeps
        # metadata filtering exact rather than losing valid filtered results.
        response = self._collection.query(
            query_texts=[query],
            n_results=len(self._documents),
            include=["distances"],
        )
        ids = response.get("ids", [[]])[0]
        distances = response.get("distances", [[]])[0]
        hits: list[tuple[str, float]] = []
        for document_id, distance in zip(ids, distances, strict=True):
            if allowed_ids is not None and document_id not in allowed_ids:
                continue
            score = 1.0 / (1.0 + max(float(distance), 0.0))
            hits.append((document_id, score))
            if len(hits) >= limit:
                break
        return hits
