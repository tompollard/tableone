"""
The tableone package is used for creating "Table 1" summary statistics for
research papers.
"""
from typing import Optional, Union
import warnings

import numpy as np
import pandas as pd
from tabulate import tabulate

from tableone.deprecations import handle_deprecated_parameters
from tableone.formatting import (docstring_copier, set_display_options, format_pvalues,
                                 format_smd_columns, apply_limits, sort_and_reindex,
                                 apply_order, mask_duplicate_values, create_row_labels,
                                 reorder_columns)
from tableone.preprocessors import (ensure_list, detect_categorical, order_categorical,
                                    get_groups, handle_categorical_nulls)
from tableone.statistics import Statistics
from tableone.tables import Tables
from tableone.validators import DataValidator, InputValidator


def load_dataset(name: str) -> pd.DataFrame:
    """
    Load an example dataset from the online repository (requires internet).

    These datasets are useful for documentation and testing.

    Parameters
    ----------
    name : str
        Name of the dataset.

    Returns
    -------
    df : :class:`pandas.DataFrame`
        Tabular data.
    """
    path = ("https://raw.githubusercontent.com/"
            "tompollard/tableone/master/datasets/{}.csv")
    full_path = path.format(name)

    df = pd.read_csv(full_path)

    return df


class TableOne:
    """

    If you use the tableone package, please cite:

    Pollard TJ, Johnson AEW, Raffa JD, Mark RG (2018). tableone: An open source
    Python package for producing summary statistics for research papers.
    JAMIA Open, Volume 1, Issue 1, 1 July 2018, Pages 26-31.
    https://doi.org/10.1093/jamiaopen/ooy012

    Create an instance of the tableone summary table.

    Parameters
    ----------
    data : pandas DataFrame
        The dataset to be summarised. Rows are observations, columns are
        variables.
    columns : list, optional
        List of columns in the dataset to be included in the final table.
        Setting the argument to None will include all columns by default.
    categorical : list, optional
        List of columns that contain categorical variables.
        If the argument is set to None (or omitted), we attempt to detect
        categorical variables. Set to an empty list to indicate explicitly
        that there are no variables of this type to be included.
    continuous : list, optional
        List of columns that contain continuous variables.
        If the argument is set to None (or omitted), we attempt to detect
        continuous variables. Set to an empty list to indicate explicitly
        that there are no variables of this type to be included.
    groupby : str, optional
        Optional column for stratifying the final table (default: None).
    nonnormal : list, optional
        List of columns that contain non-normal variables (default: None).
    min_max: list, optional
        List of variables that should report minimum and maximum, instead of
        standard deviation (for normal) or Q1-Q3 (for non-normal).
    pval : bool, optional
        Display computed P-Values (default: False).
    pval_adjust : str, optional
        Method used to adjust P-Values for multiple testing.
        The P-values from the unadjusted table (default when pval=True)
        are adjusted to account for the number of total tests that were
        performed.
        These adjustments would be useful when many variables are being
        screened to assess if their distribution varies by the variable in the
        groupby argument.
        For a complete list of methods, see documentation for statsmodels
        multipletests.
        Available methods include ::

        `None` : no correction applied.
        `bonferroni` : one-step correction
        `sidak` : one-step correction
        `holm-sidak` : step down method using Sidak adjustments
        `simes-hochberg` : step-up method (independent)
        `hommel` : closed method based on Simes tests (non-negative)

    htest_name : bool, optional
        Display a column with the names of hypothesis tests (default: False).
    htest : dict, optional
        Dictionary of custom hypothesis tests. Keys are variable names and
        values are functions. Functions must take a list of Numpy Arrays as
        the input argument and must return a test result.
        e.g. htest = {'age': myfunc}
    missing : bool, optional
        Display a count of null values (default: True).
    ddof : int, optional
        Degrees of freedom for standard deviation calculations (default: 1).
    rename : dict, optional
        Dictionary of alternative names for variables.
        e.g. `rename = {'sex':'gender', 'trt':'treatment'}`
    sort : bool or str, optional
        If `True`, sort the variables alphabetically. If a string
        (e.g. `'P-Value'`), sort by the specified column in ascending order.
        Default (`False`) retains the sequence specified in the `columns`
        argument. Currently the only columns supported are: `'Missing'`,
        `'P-Value'`, `'P-Value (adjusted)'`, and `'Test'`.
    limit : int or dict, optional
        Limit to the top N most frequent categories. If int, apply to all
        categorical variables. If dict, apply to the key (e.g. {'sex': 1}).
    order : dict, optional
        Specify an order for categorical variables. Key is the variable, value
        is a list of values in order.  {e.g. 'sex': ['f', 'm', 'other']}
    label_suffix : bool, optional
        Append summary type (e.g. "mean (SD); median [Q1,Q3], n (%); ") to the
        row label (default: True).
    decimals : int or dict, optional
        Number of decimal places to display. An integer applies the rule to all
        variables (default: 1). A dictionary (e.g. `decimals = {'age': 0)`)
        applies the rule per variable, defaulting to 1 place for unspecified
        variables. For continuous variables, applies to all summary statistics
        (e.g. mean and standard deviation). For categorical variables, applies
        to percentage only.
    overall : bool, optional
        If True, add an "overall" column to the table. Smd and p-value
        calculations are performed only using stratified columns.
    row_percent : bool, optional
        If True, compute "n (%)" percentages for categorical variables across
        "groupby" rows rather than columns.
    display_all : bool, optional
        If True, set pd. display_options to display all columns and rows.
        (default: False)
    dip_test : bool, optional
        Run Hartigan's Dip Test for multimodality. If variables are found to
        have multimodal distributions, a remark will be added below the
        Table 1.
        (default: False)
    normal_test : bool, optional
        Test the null hypothesis that a sample come from a normal distribution.
        Uses scipy.stats.normaltest. If variables are found to have non-normal
        distributions, a remark will be added below the Table 1.
        (default: False)
    tukey_test : bool, optional
        Run Tukey's test for far outliers. If variables are found to
        have far outliers, a remark will be added below the Table 1.
        (default: False)
    include_null : bool, optional
        Include None/Null values for categorical variables by treating them as a
        category level. (default: True)


    Attributes
    ----------
    tableone : dataframe
        Summary of the data (i.e., the "Table 1").

    Examples
    --------
    >>> df = pd.DataFrame({'size': [1, 2, 60, 1, 1],
    ...                   'fruit': ['peach', 'orange', 'peach', 'peach',
    ...                             'orange'],
    ...                   'tasty': ['yes', 'yes', 'no', 'yes', 'no']})

    >>> df
       size   fruit tasty
    0     1   peach   yes
    1     2  orange   yes
    2    60   peach    no
    3     1   peach   yes
    4     1  orange    no

    >>> TableOne(df, overall=False, groupby="fruit", pval=True)

                        Grouped by fruit
                                 Missing     orange        peach P-Value
    n                                             2            3
    size, mean (SD)                    0  1.5 (0.7)  20.7 (34.1)   0.433
    tasty, n (%)    no                 0   1 (50.0)     1 (33.3)   1.000
                    yes                    1 (50.0)     2 (66.7)

    ...
    """
    def __init__(self, data: pd.DataFrame,
                 columns: Optional[list] = None,
                 categorical: Optional[list] = None,
                 continuous: Optional[list] = None,
                 groupby: Optional[str] = None,
                 nonnormal: Optional[list] = None,
                 min_max: Optional[list] = None, pval: Optional[bool] = False,
                 pval_adjust: Optional[str] = None, htest_name: bool = False,
                 pval_test_name: bool = False, htest: Optional[dict] = None,
                 isnull: Optional[bool] = None, missing: bool = True,
                 ddof: int = 1, labels: Optional[dict] = None,
                 rename: Optional[dict] = None, sort: Union[bool, str] = False,
                 limit: Union[int, dict, None] = None,
                 order: Optional[dict] = None, remarks: bool = False,
                 label_suffix: bool = True, decimals: Union[int, dict] = 1,
                 smd: bool = False, overall: bool = True,
                 row_percent: bool = False, display_all: bool = False,
                 dip_test: bool = False, normal_test: bool = False,
                 tukey_test: bool = False,
                 pval_threshold: Optional[float] = None,
                 include_null: Optional[bool] = True) -> None:

        # Warn about deprecated parameters
        handle_deprecated_parameters(labels, isnull, pval_test_name, remarks)

        # Attach submodules
        self.statistics = Statistics()
        self.tables = Tables()

        # Initialize attributes
        data = self.initialize_core_attributes(data, columns, categorical, continuous, groupby,
                                               nonnormal, min_max, pval, pval_adjust, htest_name,
                                               htest, missing, ddof, rename, sort, limit, order,
                                               label_suffix, decimals, smd, overall, row_percent,
                                               dip_test, normal_test, tukey_test, pval_threshold,
                                               include_null)

        # Initialize intermediate tables
        self.initialize_intermediate_tables()

        # Set up validators and validate data
        self.setup_validators()
        self.validate_data(data)

        # Create all intermediate tables
        self.create_intermediate_tables(data)

        # Assemble Table 1
        self.tableone = self._create_tableone(data)

        # wrap dataframe methods
        self.head = self.tableone.head
        self.tail = self.tableone.tail
        self.to_csv = self.tableone.to_csv
        self.to_excel = self.tableone.to_excel
        self.to_html = self.tableone.to_html
        self.to_json = self.tableone.to_json
        self.to_latex = self.tableone.to_latex

        # set display options
        if display_all:
            set_display_options()

    def __str__(self) -> str:
        return self.tableone.to_string() + self._generate_remarks('\n')

    def __repr__(self) -> str:
        return self.tableone.to_string() + self._generate_remarks('\n')

    def _repr_html_(self) -> str:
        return self.tableone._repr_html_() + self._generate_remarks('<br />')

    def initialize_core_attributes(self, data, columns, categorical, continuous, groupby,
                                   nonnormal, min_max, pval, pval_adjust, htest_name,
                                   htest, missing, ddof, rename, sort, limit, order,
                                   label_suffix, decimals, smd, overall, row_percent, 
                                   dip_test, normal_test, tukey_test, pval_threshold,
                                   include_null):
        """
        Initialize attributes.
        """
        self._alt_labels = rename
        self._include_null = include_null
        self._columns = columns if columns else data.columns.to_list()  # type: ignore
        self._categorical = detect_categorical(data[self._columns], groupby) if categorical is None else categorical
        if continuous:
            self._continuous = continuous
        else:
            self._continuous = [c for c in self._columns if c not in self._categorical + [groupby]]  # type: ignore
        self._ddof = ddof
        self._decimals = decimals
        self._dip_test = dip_test
        self._groupby = groupby
        self._htest = htest
        self._isnull = missing
        self._label_suffix = label_suffix
        self._limit = limit
        self._min_max = min_max
        self._nonnormal = ensure_list(nonnormal, arg_name="nonnormal")  # type: ignore
        self._normal_test = normal_test
        self._order = order_categorical(data, order)
        self._overall = overall
        self._pval = pval
        self._pval_adjust = pval_adjust
        self._pval_test_name = htest_name
        self._pval_threshold = pval_threshold
        self._reserved_columns = ['Missing', 'P-Value', 'Test', 'P-Value (adjusted)', 'SMD', 'Overall']
        self._row_percent = row_percent
        self._smd = smd
        self._sort = sort
        self._tukey_test = tukey_test
        self._warnings = {}

        if self._categorical and self._include_null:
            data[self._categorical] = handle_categorical_nulls(data[self._categorical])

        self._groupbylvls = get_groups(data, self._groupby, self._order, self._reserved_columns)

        return data

    def initialize_intermediate_tables(self):
        """
        Initialize the intermediate tables.
        """
        # Intermediate tables
        self.htest_table = None
        self.cat_describe_all = None
        self.cont_describe_all = None
        self.cat_describe = None
        self.cont_describe = None
        self.smd_table = None
        self.cat_table = None
        self.cont_table = None

    def setup_validators(self):
        self.data_validator = DataValidator()
        self.input_validator = InputValidator()

    def validate_data(self, data):
        self.input_validator.validate(self._groupby, self._nonnormal, self._min_max,  # type: ignore
                                      self._pval_adjust, self._order, self._pval,  # type: ignore
                                      self._columns, self._categorical, self._continuous)  # type: ignore
        self.data_validator.validate(data, self._columns, self._categorical, self._include_null)  # type: ignore

    def create_intermediate_tables(self, data):
        """
        Creates all intermediate tables.
        """
        # forgive me jraffa
        if self._pval:
            self.htest_table = self.tables.create_htest_table(data, self._continuous, self._categorical,
                                                              self._nonnormal, self._groupby,
                                                              self._groupbylvls, self._htest,
                                                              self._pval, self._pval_adjust)

        # create overall tables if required
        if self._categorical and self._groupby and self._overall:
            self.cat_describe_all = self.tables.create_cat_describe(data,
                                                                    self._categorical,
                                                                    self._decimals,
                                                                    self._row_percent,
                                                                    self._include_null,
                                                                    groupby=None,
                                                                    groupbylvls=['Overall'])

        if self._continuous and self._groupby and self._overall:
            self.cont_describe_all = self.tables.create_cont_describe(data,
                                                                      self._ddof,
                                                                      self._t1_summary,
                                                                      self._dip_test,
                                                                      self._tukey_test,
                                                                      self._normal_test,
                                                                      self._continuous,
                                                                      groupby=None)

        # create descriptive tables
        if self._categorical:
            self.cat_describe = self.tables.create_cat_describe(data,
                                                                self._categorical,
                                                                self._decimals,
                                                                self._row_percent,
                                                                self._include_null,
                                                                groupby=self._groupby,
                                                                groupbylvls=self._groupbylvls)

        if self._continuous:
            self.cont_describe = self.tables.create_cont_describe(data,
                                                                  self._ddof,
                                                                  self._t1_summary,
                                                                  self._dip_test,
                                                                  self._tukey_test,
                                                                  self._normal_test,
                                                                  self._continuous,
                                                                  groupby=self._groupby)

        # compute standardized mean differences
        if self._smd:
            self.smd_table = self.tables.create_smd_table(self._groupbylvls,
                                                          self._continuous,
                                                          self._categorical,
                                                          self.cont_describe,
                                                          self.cat_describe)

        # create continuous and categorical tables
        if self._categorical:
            self.cat_table = self.tables.create_cat_table(data,
                                                          self._overall,
                                                          self.cat_describe,
                                                          self._categorical,
                                                          self._include_null,
                                                          self._pval,
                                                          self._pval_adjust,
                                                          self.htest_table,
                                                          self._smd,
                                                          self.smd_table,
                                                          self._groupby,
                                                          self.cat_describe_all)

        if self._continuous:
            self.cont_table = self.tables.create_cont_table(data,
                                                            self._overall,
                                                            self.cont_describe,
                                                            self.cont_describe_all,
                                                            self._continuous,
                                                            self._pval,
                                                            self._pval_adjust,
                                                            self.htest_table,
                                                            self._smd,
                                                            self.smd_table,
                                                            self._groupby)

    def tabulate(self, headers=None, tablefmt='grid', **kwargs) -> str:
        """
        Pretty-print tableone data. Wrapper for the Python 'tabulate' library.

        Args:
            headers (list): Defines a list of column headers to be used.
            tablefmt (str): Defines how the table is formatted. Table formats
                include: 'plain','simple','github','grid','fancy_grid','pipe',
                'orgtbl','jira','presto','psql','rst','mediawiki','moinmoin',
                'youtrack','html','latex','latex_raw','latex_booktabs',
                and 'textile'.

        Examples:
            To output tableone in github syntax, call tabulate with the
                'tablefmt="github"' argument.

            >>> print(tableone.tabulate(tablefmt='fancy_grid'))
        """
        # reformat table for tabulate
        df = self.tableone

        if not headers:
            try:
                headers = df.columns.levels[1]
            except AttributeError:
                headers = df.columns

        df = df.reset_index()
        df = df.set_index('level_0')
        isdupe = df.index.duplicated()
        df.index = df.index.where(~isdupe, '')
        df = df.rename_axis(None).rename(columns={'level_1': ''})

        return tabulate(df, headers=headers, tablefmt=tablefmt, **kwargs)

    def _generate_remarks(self, newline='\n') -> str:
        """
        Generate a series of remarks that the user should consider
        when interpreting the summary statistics.
        """
        if self.cont_describe is not None:
            # generate warnings for continuous variables
            if self._continuous and self._tukey_test:
                # highlight far outliers
                outlier_mask = self.cont_describe.far_outliers > 1
                outlier_vars = list(self.cont_describe.far_outliers[outlier_mask].
                                    dropna(how='all').index)
                if outlier_vars:
                    self._warnings["""Tukey test indicates far outliers
                                    in"""] = outlier_vars

            if self._continuous and self._dip_test:
                # highlight possible multimodal distributions using hartigan's dip
                # test -1 values indicate NaN
                modal_mask = ((self.cont_describe.hartigan_dip >= 0) &
                            (self.cont_describe.hartigan_dip <= 0.05))
                modal_vars = list(self.cont_describe.hartigan_dip[modal_mask].
                                dropna(how='all').index)
                if modal_vars:
                    self._warnings["""Hartigan's Dip Test reports possible
                                    multimodal distributions for"""] = modal_vars

            if self._continuous and self._normal_test:
                # highlight non normal distributions
                # -1 values indicate NaN
                modal_mask = ((self.cont_describe.normality >= 0) &
                            (self.cont_describe.normality <= 0.001))
                modal_vars = list(self.cont_describe.normality[modal_mask].
                                dropna(how='all').index)
                if modal_vars:
                    self._warnings["""Normality test reports non-normal
                                    distributions for"""] = modal_vars

            # create the warning string
            msg = '{}'.format(newline)
            for n, k in enumerate(sorted(self._warnings)):
                msg += '[{}] {}: {}.{}'.format(n+1, k, ', '.join(self._warnings[k]), newline)
        else:
            msg = ""

        return msg

    def _t1_summary(self, x: pd.Series) -> str:
        """
        Compute median [IQR] or mean (Std) for the input series.

        Parameters
        ----------
            x : pandas Series
                Series of values to be summarised.
        """
        # set decimal places
        if isinstance(self._decimals, int):
            n = self._decimals
        elif isinstance(self._decimals, dict):
            try:
                n = self._decimals[x.name]
            except KeyError:
                n = 1
        else:
            n = 1
            msg = """The decimals arg must be an int or dict.
                     Defaulting to {} d.p.""".format(n)
            warnings.warn(msg)

        if x.name in self._nonnormal:
            f = "{{:.{}f}} [{{:.{}f}},{{:.{}f}}]".format(n, n, n)
            if self._min_max and x.name in self._min_max:
                return f.format(
                    np.nanmedian(x.values), np.nanmin(x.values),  # type: ignore
                    np.nanmax(x.values),  # type: ignore
                )
            else:
                return f.format(
                    np.nanmedian(x.values),  # type: ignore
                    np.nanpercentile(x.values, 25),  # type: ignore
                    np.nanpercentile(x.values, 75),  # type: ignore
                )
        else:
            if self._min_max and x.name in self._min_max:
                f = "{{:.{}f}} [{{:.{}f}},{{:.{}f}}]".format(n, n, n)
                return f.format(
                    np.nanmean(x.values), np.nanmin(x.values),  # type: ignore
                    np.nanmax(x.values),  # type: ignore
                )
            else:
                f = '{{:.{}f}} ({{:.{}f}})'.format(n, n)
                return f.format(np.nanmean(x.values), self.statistics._std(x, self._ddof))  # type: ignore

    def _combine_tables(self):
        """
        Combine continuous and categorical tables.
        """
        table = pd.DataFrame()
        if self._continuous and self._categorical:
            # support pandas<=0.22
            try:
                table = pd.concat([self.cont_table, self.cat_table], sort=False)
            except TypeError:
                table = pd.concat([self.cont_table, self.cat_table])
        elif self._continuous:
            table = self.cont_table
        elif self._categorical:
            table = self.cat_table

        return table

    def _insert_n_row(self, table, data):
        """
        Inserts a row that shows 'n', the total number or count of items
        within each group or overall.
        """
        # insert n row
        n_row = pd.DataFrame(columns=['variable', 'value', 'Missing'])
        n_row = n_row.set_index(['variable', 'value'])
        n_row.loc['n', 'Missing'] = None

        # support pandas<=0.22
        try:
            table = pd.concat([n_row, table], sort=False)
        except TypeError:
            table = pd.concat([n_row, table])

        if self._groupbylvls == ['Overall']:
            table.loc['n', 'Overall'] = len(data.index)
        else:
            if self._overall:
                table.loc['n', 'Overall'] = len(data.index)
            for g in self._groupbylvls:
                ct = data[self._groupby][data[self._groupby] == g].count()
                table.loc['n', '{}'.format(g)] = ct

        return table

    def _apply_alt_labels(self, table):
        """
        Applies alternative labels to the variables if required.
        """
        # display alternative labels if assigned
        table = table.rename(index=create_row_labels(self._columns,
                                                     self._alt_labels,
                                                     self._label_suffix,
                                                     self._nonnormal,
                                                     self._min_max,
                                                     self._categorical
                                                     ), level=0)

        return table

    def _add_groupby_columns(self, table):
        """
        Adds multi-level column headers to denote grouping by a specific
        variable, for clarity.
        """
        # add column index
        if not self._groupbylvls == ['Overall']:
            # rename groupby variable if requested
            c = self._groupby
            if self._alt_labels:
                if self._groupby in self._alt_labels:
                    c = self._alt_labels[self._groupby]

            c = 'Grouped by {}'.format(c)
            table.columns = pd.MultiIndex.from_product([[c], table.columns])

        return table

    def _create_tableone(self, data):
        """
        Create table 1 by combining the continuous and categorical tables.

        Returns
        ----------
        table : pandas DataFrame
            The complete table one.
        """
        table = self._combine_tables()
        optional_columns = ['P-Value', 'P-Value (adjusted)', 'Test']

        # ensure column headers are strings before reindexing
        table = table.reset_index().set_index(['variable', 'value'])  # type: ignore
        table.columns = table.columns.values.astype(str)

        table = sort_and_reindex(table, self._smd, self.smd_table, self._sort, self._columns)
        table = format_pvalues(table, self._pval, self._pval_adjust, self._pval_threshold)
        table = format_smd_columns(table, self._smd, self.smd_table)
        table = apply_order(table, self._order, self._groupby)
        table = apply_limits(table, data, self._limit, self._categorical, self._order)
        table = self._insert_n_row(table, data)
        table = mask_duplicate_values(table, optional_columns, self._smd, self.smd_table)

        # remove unwanted columns
        if not self._isnull:
            table = table.drop('Missing', axis=1)

        if self._pval and not self._pval_test_name:
            table = table.drop('Test', axis=1)

        # replace nans with empty strings
        table = table.fillna('')

        table = self._add_groupby_columns(table)
        table = self._apply_alt_labels(table)
        table = reorder_columns(table, optional_columns, self._groupby, self._order, self._overall)

        try:
            if 'Missing' in self._alt_labels or 'Overall' in self._alt_labels:  # type: ignore
                table = table.rename(columns=self._alt_labels)
        except TypeError:
            pass

        # remove the 'variable, value' column names in the index
        table = table.rename_axis([None, None])

        return table


# Allow TableOne to be called as a function.
# Refactor this out at some point!
@docstring_copier(TableOne.__doc__)
def tableone(*args, **kwargs):
    """{0}"""
    return TableOne(*args, **kwargs)
