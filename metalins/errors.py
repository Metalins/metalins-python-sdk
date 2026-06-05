"""Errors raised by the metalins SDK."""


class MetalinsError(Exception):
    """Base class for all SDK errors. Raised directly for generic 4xx
    responses and unexpected status codes."""


class AuthenticationError(MetalinsError):
    """Raised when the API key is missing or invalid (HTTP 401)."""


class AgentNotFound(MetalinsError):
    """Raised when the agent_id doesn't exist or is revoked (HTTP 404)."""


class ServerError(MetalinsError):
    """Raised when the server returns 5xx."""
