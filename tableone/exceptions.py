"""Exceptions and warnings"""
import warnings


class InputError(Exception):
    """Custom exception for input validation errors."""
    pass

def non_continuous_warning(c):
    msg = ("'{}' has all non-numeric values. Consider including "
           "it in the list of categorical variables.").format(c)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)
