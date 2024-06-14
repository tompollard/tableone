from typing import Any, List, Optional, Union, Dict

import pandas as pd

from tableone.exceptions import InputError


class DataValidator:
    def __init__(self):
        """Initialize the DataValidator class."""
        pass

    def validate(self, data: pd.DataFrame, columns: list,
                 categorical: list,
                 include_null: bool) -> None:
        """
        Check the input dataset for obvious issues.

        Parameters:
        data (pd.DataFrame): The input dataframe for validation.
        columns (list): A list of columns expected to be in the dataframe.
        """
        self.check_empty_data(data)
        self.check_unique_index(data)
        self.check_columns_exist(data, columns)
        self.check_duplicate_columns(data, columns)

    def validate_input(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise InputError("Data must be a pandas DataFrame")

    def check_empty_data(self, data: pd.DataFrame):
        """Ensure the dataframe is not empty."""
        if data.empty:
            raise InputError("Input data is empty.")

    def check_unique_index(self, data: pd.DataFrame):
        """Ensure the dataframe's index is unique."""
        if not data.index.is_unique:
            raise InputError("Input data contains duplicate values in the "
                             "index. Reset the index and try again.")

    def check_columns_exist(self, data: pd.DataFrame, columns: list):
        """Ensure all required columns are present in the dataframe."""
        if not set(columns).issubset(data.columns):  # type: ignore
            missing_cols = list(set(columns) - set(data.columns))  # type: ignore
            raise InputError("""The following columns were not found in the
                                dataset: {}""".format(missing_cols))

    def check_duplicate_columns(self, data: pd.DataFrame, columns: list):
        """Ensure no duplicate columns in the data."""
        dups = data[columns].columns[
            data[columns].columns.duplicated()].unique()
        if not dups.empty:
            raise InputError("""Input data contains duplicate
                                columns: {}""".format(dups))


class InputValidator:
    def __init__(self):
        """Initialize the InputValidator class."""
        pass

    def validate(self,
                 groupby: str,
                 nonnormal: Union[List[str], str],
                 min_max: Union[List[str], str],
                 pval_adjust: str,
                 order: Dict[str, List[Any]],
                 pval: bool,
                 columns: List[str],
                 categorical: List[str],
                 continuous: List[str]) -> None:
        """
        Check the input dataset for obvious issues.

        Parameters:
        data (pd.DataFrame): The input dataframe for validation.
        columns (list): A list of columns expected to be in the dataframe.
        """
        self.check_groupby(groupby, pval)
        self.check_list(nonnormal, 'nonnormal')
        self.check_list(min_max, 'min_max', expected_type=str)
        self.check_pval_adjust(pval_adjust)
        self.check_order(order)
        self.check_exclusivity(categorical, continuous)
        self.check_columns_exist(columns, categorical, continuous)

    def check_groupby(self, groupby: str, pval: bool) -> None:
        """Ensure 'groupby' is provided as a str."""
        if groupby:
            if isinstance(groupby, list):
                msg = (f"Invalid 'groupby' type: expected a string, received a list. "
                       f"Use '{groupby[0]}' if it's the intended group.")
                raise ValueError(msg)
            elif not isinstance(groupby, str):
                msg = f"Invalid 'groupby' type: expected a string, received {type(groupby).__name__}."
                raise TypeError(msg)
        elif pval:
            msg = "The 'pval' parameter is set to True, but no 'groupby' parameter was specified."
            raise ValueError(msg)

    def check_list(self,
                   parameter: Optional[Union[List[Any], str]],
                   parameter_name: str,
                   expected_type: Optional[type] = None) -> None:
        """Ensure list arguments are properly formatted."""
        if parameter:
            if not isinstance(parameter, (list, str)):
                msg = (f"Invalid '{parameter_name}' type: expected a list "
                       f"or a string, received {type(parameter).__name__}.")
                raise TypeError(msg)
            if expected_type and any(not isinstance(item, expected_type) for item in parameter):
                msg = f"All items in '{parameter_name}' list must be of type {expected_type.__name__}."
                raise ValueError(msg)

    def check_pval_adjust(self, pval_adjust: str):
        """Ensure 'pval_adjust' is a known method."""
        if pval_adjust is not None:
            valid_methods = {"bonferroni", "sidak", "holm-sidak", "simes-hochberg", "hommel", None}
            if isinstance(pval_adjust, str):
                if pval_adjust.lower() not in valid_methods:
                    msg = (f"Invalid 'pval_adjust' value: '{pval_adjust}'. "
                           f"Expected one of {', '.join(valid_methods)} or None.")
                    raise ValueError(msg)
            else:
                msg = (f"Invalid type for 'pval_adjust': expected a string or None, "
                       f"received {type(pval_adjust).__name__}.")
                raise TypeError(msg)

    def check_order(self, order: dict):
        """Ensure the order argument is correctly specified."""
        if order is not None:
            if not isinstance(order, dict):
                msg = ("The 'order' parameter must be a dictionary where keys are "
                       "column names and values are lists of ordered categories.")
                raise TypeError(msg)

            for key, values in order.items():
                if not isinstance(values, list):
                    msg = f"The value for '{key}' in 'order' must be a list of categories."
                    raise TypeError(msg)

    def check_exclusivity(self, categorical: list, continuous: list):
        """Ensure categorical and continuous are mutually exclusive."""
        if categorical is None:
            categorical = []
        if continuous is None:
            continuous = []

        if set(categorical) & set(continuous):
            msg = ("Columns cannot be both categorical and continuous: "
                   f"{set(categorical) & set(continuous)}")
            raise ValueError(msg)

    def check_columns_exist(self, columns: list, categorical: list, continuous: list):
        """Ensure all specified columns exist in the DataFrame columns list."""
        if categorical:
            cat_set = set(categorical)
        else:
            cat_set = set()

        if continuous:
            cont_set = set(continuous)
        else:
            cont_set = set()

        all_specified = cat_set.union(cont_set)
        if not all_specified.issubset(set(columns)):
            missing = list(all_specified - set(columns))
            msg = f"Specified categorical/continuous columns not found in the DataFrame: {missing}"
            raise ValueError(msg)
