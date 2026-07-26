"""Exceptions with safe, API-facing semantics."""


class OrchestrationError(RuntimeError):
    """Base exception for an orchestration request failure."""


class BackendUnavailableError(OrchestrationError):
    """The requested backend is not installed or configured."""


class ModelConfigurationError(BackendUnavailableError):
    """Required model configuration is missing."""


class ModelInvocationError(OrchestrationError):
    """The configured model provider rejected or failed a request."""


class RetrievalError(OrchestrationError):
    """The MCP retrieval service could not fulfill a request."""
