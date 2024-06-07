

def ensure_list(arg, arg_name):
    """
    Ensure input argument is a list.
    """
    if arg is None:
        return []
    elif isinstance(arg, str):
        return [arg]
    elif isinstance(arg, list):
        return arg
    else:
        raise TypeError(f"{arg_name} must be a string or a list of strings.")
