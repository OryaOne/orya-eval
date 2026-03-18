"""Project-specific exceptions."""


class OryaEvalError(Exception):
    """Base exception for user-facing CLI errors."""


class ConfigError(OryaEvalError):
    """Raised when an evaluation config is invalid."""


class DataError(OryaEvalError):
    """Raised when evaluation data cannot be loaded or validated."""


class ComparisonError(OryaEvalError):
    """Raised when result comparison cannot be completed."""
