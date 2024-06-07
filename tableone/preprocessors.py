import numpy as np

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


def detect_categorical(data, groupby) -> list:
    """
    Detect categorical columns if they are not specified.

    Parameters
    ----------
        data : pandas DataFrame
            The input dataset.
        groupby : str (optional)
            The groupby variable.

    Returns
    ----------
        likely_cat : list
            List of variables that appear to be categorical.
    """
    # assume all non-numerical and date columns are categorical
    numeric_cols = set(data._get_numeric_data().columns.values)
    date_cols = set(data.select_dtypes(include=[np.datetime64]).columns)
    likely_cat = set(data.columns) - numeric_cols
    likely_cat = list(likely_cat - date_cols)

    # check proportion of unique values if numerical
    for var in data._get_numeric_data().columns:
        likely_flag = 1.0 * data[var].nunique()/data[var].count() < 0.005
        if likely_flag:
            likely_cat.append(var)

    if groupby:
        likely_cat = [x for x in likely_cat if x != groupby]

    return likely_cat
