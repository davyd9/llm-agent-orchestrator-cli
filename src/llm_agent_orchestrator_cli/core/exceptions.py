import typing as t

class LLMOrchestratorError(Exception):
    """Base exception for all application-specific errors."""
    def __init__(self, message: str, original_exception: t.Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception

    def __str__(self) -> str:
        if self.original_exception:
            return f"{super().__str__()} -> Caused by: {self.original_exception.__class__.__name__}: {self.original_exception}"
        return super().__str__()

class ConfigurationError(LLMOrchestratorError):
    """Raised for errors related to application configuration."""
    pass

class APIKeyNotFoundError(ConfigurationError):
    """Raised when a required API key is not found in the environment variables."""
    def __init__(self, env_var_name: str):
        message = f"API key environment variable '{env_var_name}' not found. Please set it in your .env file or environment."
        super().__init__(message)
        self.env_var_name = env_var_name

class ProviderError(LLMOrchestratorError):
    """Raised for errors originating from an LLM provider's API."""
    pass

class APIConnectionError(ProviderError):
    """Raised when there is a problem connecting to the LLM provider's API."""
    pass

class RateLimitError(ProviderError):
    """Raised when the API rate limit has been exceeded."""
    pass

class AuthenticationError(ProviderError):
    """Raised for authentication failures with the provider's API."""
    pass

class InvalidRequestError(ProviderError):
    """Raised when the provider's API rejects the request as invalid or malformed."""
    pass
