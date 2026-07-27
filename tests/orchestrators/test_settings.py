from __future__ import annotations

import unittest

from connoisseur.orchestrators.settings import AppSettings


class AppSettingsTests(unittest.TestCase):
    def test_reads_shared_deployment_environment(self) -> None:
        settings = AppSettings.from_env(
            {
                "LLM_MODEL": "provider/model",
                "OPENAI_API_KEY": "test-only",
                "OPENAI_BASE_URL": "http://model.test/v1",
                "MCP_SERVER_URL": "http://mcp:8001/mcp",
                "API_BASE_URL": "http://api:8000/",
                "RETRIEVAL_TOP_K": "7",
                "LANGGRAPH_MAX_TOOL_ROUNDS": "4",
                "LLM_MAX_TOKENS_PARAMETER": "max_tokens",
                "LLM_REASONING_EFFORT": "none",
                "CLIENT_TIMEOUT_SECONDS": "420",
            }
        )

        self.assertEqual(settings.llm_model, "provider/model")
        self.assertEqual(settings.openai_base_url, "http://model.test/v1")
        self.assertEqual(settings.mcp_server_url, "http://mcp:8001/mcp")
        self.assertEqual(settings.api_base_url, "http://api:8000")
        self.assertEqual(settings.retrieval_limit, 7)
        self.assertEqual(settings.langgraph_max_tool_rounds, 4)
        self.assertEqual(settings.llm_max_tokens_parameter, "max_tokens")
        self.assertEqual(settings.llm_reasoning_effort, "none")
        self.assertEqual(settings.client_timeout_seconds, 420)
        self.assertIsNone(settings.model_configuration_error())

    def test_reports_missing_model_configuration_without_values(self) -> None:
        settings = AppSettings.from_env({})

        reason = settings.model_configuration_error()

        self.assertIsNotNone(reason)
        self.assertIn("LLM_MODEL", reason or "")
        self.assertIn("OPENAI_API_KEY", reason or "")

    def test_allows_keyless_openai_compatible_local_endpoint(self) -> None:
        settings = AppSettings.from_env(
            {
                "LLM_MODEL": "local-model",
                "OPENAI_BASE_URL": "http://localhost:11434/v1",
            }
        )

        self.assertIsNone(settings.model_configuration_error())
        self.assertEqual(settings.openai_client_api_key, "not-required")

    def test_accepts_legacy_retrieval_limit_alias(self) -> None:
        settings = AppSettings.from_env({"RETRIEVAL_LIMIT": "9"})

        self.assertEqual(settings.retrieval_limit, 9)

    def test_retrieval_top_k_takes_precedence_over_limit(self) -> None:
        settings = AppSettings.from_env(
            {
                "RETRIEVAL_TOP_K": "6",
                "RETRIEVAL_LIMIT": "9",
            }
        )

        self.assertEqual(settings.retrieval_limit, 6)

    def test_rejects_invalid_numeric_settings(self) -> None:
        with self.assertRaises(ValueError):
            AppSettings.from_env({"RETRIEVAL_TOP_K": "0"})
        with self.assertRaises(ValueError):
            AppSettings.from_env({"RETRIEVAL_TOP_K": "51"})
        with self.assertRaises(ValueError):
            AppSettings.from_env({"MAX_HISTORY_MESSAGES": "101"})
        with self.assertRaises(ValueError):
            AppSettings.from_env({"LANGGRAPH_MAX_TOOL_ROUNDS": "0"})
        with self.assertRaises(ValueError):
            AppSettings.from_env({"LANGGRAPH_MAX_TOOL_ROUNDS": "11"})

    def test_rejects_unknown_max_tokens_parameter(self) -> None:
        with self.assertRaises(ValueError):
            AppSettings.from_env({"LLM_MAX_TOKENS_PARAMETER": "unknown"})

    def test_rejects_unknown_reasoning_effort(self) -> None:
        with self.assertRaises(ValueError):
            AppSettings.from_env({"LLM_REASONING_EFFORT": "extreme"})
