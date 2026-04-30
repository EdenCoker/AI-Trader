class BrokerError(RuntimeError):
    """Raised for broker execution failures."""


class BrokerConfigurationError(BrokerError):
    """Raised when broker config is invalid or unsafe."""


class BrokerConnectionError(BrokerError):
    """Raised when broker connection fails."""

