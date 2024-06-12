import warnings


def handle_deprecated_parameters(labels, isnull, pval_test_name, remarks):
    """
    Raise warnings for deprecated parameters.
    """
    deprecated_parameter(labels, "labels", "Use 'rename' instead")
    deprecated_parameter(isnull, "isnull", "Use 'missing' instead")
    deprecated_parameter(pval_test_name, "pval_test_name", "Use 'htest_name' instead")
    deprecated_parameter(remarks, "remarks", "Use test names instead (e.g. diptest = True)")


def deprecated_parameter(parameter, parameter_name, message, version=None):
    """
    Raise warning for a deprecated parameter.

    Parameters:
    - parameter: parameter to be removed.
    - parameter: name of parameter to be removed.
    - message: str, message to insert into deprecated warning.
    - version: str, version when the parameter was deprecated (optional).

    Usage:
    deprecated_parameter(parameter=label, parameter_name='level, new_name='rename', version='0.7')
    """
    if parameter:
        if version:
            version_message = f"version {version}"
        else:
            version_message = "a future version"
        warnings.warn(f"{parameter_name} is deprecated and will be removed in {version_message}. {message}",
                      category=DeprecationWarning, stacklevel=2)
