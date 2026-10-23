class DataError(RuntimeError):
    """Raised when market data cannot be downloaded or fails causal checks."""


class ModelNotFoundError(FileNotFoundError):
    """Raised when a ticker has no saved checkpoint yet."""
