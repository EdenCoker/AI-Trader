class LLMError(RuntimeError):
    """Raised when an LLM request fails."""


class LLMConfigurationError(LLMError):
    """Raised when an LLM client cannot be configured from settings."""

