"""
The tableone package is used for creating "Table 1" summary statistics for
research papers.
"""
from typing import Callable, Optional, Union
import warnings

import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
from scipy import stats
from statsmodels.stats import multitest
from tabulate import tabulate

from tableone.modality import hartigan_diptest

# display deprecation warnings
warnings.simplefilter('always', DeprecationWarning)


def load_dataset(name: str):
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


def docstring_copier(*sub):
    """
    Wrap the TableOne docstring (not ideal :/)
    """
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*sub)
        return obj
    return dec


class InputError(Exception):
    """
    Exception raised for errors in the input.
    """
    pass


class TableOne(object):
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
    categorical : list, optional
        List of columns that contain categorical variables.
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
        are adjusted to account for the number of total tests that were performed.
        These adjustments would be useful when many variables are being screened
        to assess if their distribution varies by the variable in the groupby argument.
        For a complete list of methods, see documentation for statsmodels multipletests.
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
    remarks : bool, optional
        Add remarks on the appropriateness of the summary measures and the
        statistical tests (default: True).
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
    overall : bool:
        If True, add an "overall" column to the table. Smd and p-value
        calculations are performed only using stratified columns.
    display_all : bool:
        If True, set pd. display_options to display all columns and rows.
        (default: False)

    Attributes
    ----------
    tableone : dataframe
        Summary of the data (i.e., the "Table 1").

    Examples
    --------
    >>> df = pd.DataFrame({'size': [1, 2, 60, 1, 1],
    ...                   'fruit': ['peach', 'orange', 'peach', 'peach', 'orange'],
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
    def __init__(self, data: pd.DataFrame, columns: Optional[list] = None,
                 categorical: Optional[list] = None,
                 groupby: Optional[str] = None,
                 nonnormal: Optional[list] = None,
                 min_max: Optional[list] = None, pval: Optional[bool] = False,
                 pval_adjust: Optional[str] = None, htest_name: bool = False,
                 pval_test_name: bool = False, htest: Optional[dict] = None,
                 isnull: Optional[bool] = None, missing: bool = True,
                 ddof: int = 1, labels: Optional[dict] = None,
                 rename: Optional[dict] = None, sort: Union[bool, str] = False,
                 limit: Union[int, dict, None] = None,
                 order: Optional[dict] = None, remarks: bool = True,
                 label_suffix: bool = True, decimals: Union[int, dict] = 1,
                 smd: bool = False, overall: bool = True,
                 display_all: bool = False) -> None:

        # labels is now rename
        if labels is not None and rename is not None:
            raise TypeError("TableOne received both labels and rename.")
        elif labels is not None:
            warnings.warn("The labels argument is deprecated; use "
                          "rename instead.", DeprecationWarning)
            self._alt_labels = labels
        else:
            self._alt_labels = rename

        # isnull is now missing
        if isnull is not None:
            warnings.warn("The isnull argument is deprecated; use "
                          "missing instead.", DeprecationWarning)
            self._isnull = isnull
        else:
            self._isnull = missing

        # pval_test_name is now htest_name
        if pval_test_name:
            warnings.warn("The pval_test_name argument is deprecated; use "
                          "htest_name instead.", DeprecationWarning)
            self._pval_test_name = pval_test_name
        else:
            self._pval_test_name = htest_name

        # groupby should be a string
        if not groupby:
            groupby = ''
        elif groupby and type(groupby) == list:
            groupby = groupby[0]

        # nonnormal should be a string
        if not nonnormal:
            nonnormal = []
        elif nonnormal and type(nonnormal) == str:
            nonnormal = [nonnormal]

        # min_max should be a list
        if min_max and isinstance(min_max, bool):
            warnings.warn("min_max should specify a list of variables.")
            min_max = None

        # if the input dataframe is empty, raise error
        if data.empty:
            raise InputError("Input data is empty.")

        # if the input dataframe has a non-unique index, raise error
        if not data.index.is_unique:
            raise InputError("Input data contains duplicate values in the "
                             "index. Reset the index and try again.")

        # if columns are not specified, use all columns
        if not columns:
            columns = data.columns.values

        # check that the columns exist in the dataframe
        if not set(columns).issubset(data.columns):
            notfound = list(set(columns) - set(data.columns))
            raise InputError("""Columns not found in
                                dataset: {}""".format(notfound))

        # check for duplicate columns
        dups = data[columns].columns[data[columns].columns.duplicated()].unique()
        if not dups.empty:
            raise InputError("""Input data contains duplicate
                                columns: {}""".format(dups))

        # if categorical not specified, try to identify categorical
        if not categorical and type(categorical) != list:
            categorical = self._detect_categorical_columns(data[columns])
            # omit categorical row if it is specified in groupby
            if groupby:
                categorical = [x for x in categorical if x != groupby]

        if isinstance(pval_adjust, bool) and pval_adjust:
            msg = ("pval_adjust expects a string, but a boolean was specified."
                   " Defaulting to the 'bonferroni' correction.")
            warnings.warn(msg)
            pval_adjust = "bonferroni"

        # if custom order is provided, ensure that values are strings
        if order:
            order = {k: ["{}".format(v) for v in order[k]] for k in order}

        # if input df has ordered categorical variables, get the order.
        order_cats = [x for x in data.select_dtypes("category")
                      if data[x].dtype.ordered]
        if any(order_cats):
            d_order_cats = {v: data[v].cat.categories for v in order_cats}
            d_order_cats = {k: ["{}".format(v) for v in d_order_cats[k]]
                            for k in d_order_cats}

        # combine the orders. custom order takes precedence.
        if order_cats and order:
            new = {**order, **d_order_cats}
            for k in order:
                new[k] = order[k] + [x for x in new[k] if x not in order[k]]
            order = new
        elif order_cats:
            order = d_order_cats

        if pval and not groupby:
            raise InputError("If pval=True then groupby must be specified.")

        self._columns = list(columns)
        self._continuous = [c for c in columns
                            if c not in categorical + [groupby]]
        self._categorical = categorical
        self._nonnormal = nonnormal
        self._min_max = min_max
        self._pval = pval
        self._pval_adjust = pval_adjust
        self._htest = htest
        self._sort = sort
        self._groupby = groupby
        # degrees of freedom for standard deviation
        self._ddof = ddof
        self._limit = limit
        self._order = order
        self._remarks = remarks
        self._label_suffix = label_suffix
        self._decimals = decimals
        self._smd = smd
        self._overall = overall

        # display notes and warnings below the table
        self._warnings = {}

        # output column names that cannot be contained in a groupby
        self._reserved_columns = ['Missing', 'P-Value', 'Test',
                                  'P-Value (adjusted)', 'SMD', 'Overall']

        if self._groupby:
            self._groupbylvls = sorted(data.groupby(groupby).groups.keys())

            # reorder the groupby levels if order is provided
            if self._order and self._groupby in self._order:
                unordered = [x for x in self._groupbylvls
                             if x not in self._order[self._groupby]]
                self._groupbylvls = self._order[self._groupby] + unordered

            # check that the group levels do not include reserved words
            for level in self._groupbylvls:
                if level in self._reserved_columns:
                    raise InputError("""Group level contains '{}', a reserved
                                        keyword.""".format(level))
        else:
            self._groupbylvls = ['Overall']

        # forgive me jraffa
        if self._pval:
            self._htest_table = self._create_htest_table(data)

        # correct for multiple testing
        if self._pval and self._pval_adjust:
            alpha = 0.05
            adjusted = multitest.multipletests(self._htest_table['P-Value'],
                                               alpha=alpha,
                                               method=self._pval_adjust)
            self._htest_table['P-Value (adjusted)'] = adjusted[1]
            self._htest_table['adjust method'] = self._pval_adjust

        # create overall tables if required
        if self._categorical and self._groupby and overall:
            self.cat_describe_all = self._create_cat_describe(data, False,
                                                              ['Overall'])

        if self._continuous and self._groupby and overall:
            self.cont_describe_all = self._create_cont_describe(data, False)

        # create descriptive tables
        if self._categorical:
            self.cat_describe = self._create_cat_describe(data, self._groupby,
                                                          self._groupbylvls)

        if self._continuous:
            self.cont_describe = self._create_cont_describe(data,
                                                            self._groupby)

        # compute standardized mean differences
        if self._smd:
            self.smd_table = self._create_smd_table(data)

        # create continuous and categorical tables
        if self._categorical:
            self.cat_table = self._create_cat_table(data, overall)

        if self._continuous:
            self.cont_table = self._create_cont_table(data, overall)

        # combine continuous variables and categorical variables into table 1
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
            self._set_display_options()

    def __str__(self):
        return self.tableone.to_string() + self._generate_remarks('\n')

    def __repr__(self):
        return self.tableone.to_string() + self._generate_remarks('\n')

    def _repr_html_(self):
        return self.tableone._repr_html_() + self._generate_remarks('<br />')

    def _set_display_options(self):
        """
        Set pandas display options. Display all rows and columns by default.
        """
        display_options = {'display.max_rows': None,
                           'display.max_columns': None,
                           'display.width': None,
                           'display.max_colwidth': None}

        for k in display_options:
            try:
                pd.set_option(k, display_options[k])
            except ValueError:
                msg = """Newer version of Pandas required to set the '{}'
                         option.""".format(k)
                warnings.warn(msg)

    def tabulate(self, headers=None, tablefmt='grid', **kwargs):
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

    def _generate_remarks(self, newline='\n'):
        """
        Generate a series of remarks that the user should consider
        when interpreting the summary statistics.
        """
        # generate warnings for continuous variables
        if self._continuous:
            # highlight far outliers
            outlier_mask = self.cont_describe.far_outliers > 1
            outlier_vars = list(self.cont_describe.far_outliers[outlier_mask].
                                dropna(how='all').index)
            if outlier_vars:
                self._warnings["""Tukey test indicates far outliers
                                  in"""] = outlier_vars

            # highlight possible multimodal distributions using hartigan's dip
            # test -1 values indicate NaN
            modal_mask = ((self.cont_describe.diptest >= 0) &
                          (self.cont_describe.diptest <= 0.05))
            modal_vars = list(self.cont_describe.diptest[modal_mask].
                              dropna(how='all').index)
            if modal_vars:
                self._warnings["""Hartigan's Dip Test reports possible
                                  multimodal distributions for"""] = modal_vars

            # highlight non normal distributions
            # -1 values indicate NaN
            modal_mask = ((self.cont_describe.normaltest >= 0) &
                          (self.cont_describe.normaltest <= 0.001))
            modal_vars = list(self.cont_describe.normaltest[modal_mask].
                              dropna(how='all').index)
            if modal_vars:
                self._warnings["""Normality test reports non-normal
                                  distributions for"""] = modal_vars

        # create the warning string
        msg = '{}'.format(newline)
        for n, k in enumerate(sorted(self._warnings)):
            msg += '[{}] {}: {}.{}'.format(n+1, k,
                                           ', '.join(self._warnings[k]),
                                           newline)

        return msg

    def _detect_categorical_columns(self, data):
        """
        Detect categorical columns if they are not specified.

        Parameters
        ----------
            data : pandas DataFrame
                The input dataset.

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
        return likely_cat

    def _cont_smd(self, data1=None, data2=None, mean1=None, mean2=None,
                  sd1=None, sd2=None, n1=None, n2=None, unbiased=False):
        """
        Compute the standardized mean difference (regular or unbiased) using
        either raw data or summary measures.

        Parameters
        ----------
        data1 : list
            List of values in dataset 1 (control).
        data2 : list
            List of values in dataset 2 (treatment).
        mean1 : float
            Mean of dataset 1 (control).
        mean2 : float
            Mean of dataset 2 (treatment).
        sd1 : float
            Standard deviation of dataset 1 (control).
        sd2 : float
            Standard deviation of dataset 2 (treatment).
        n1 : int
            Sample size of dataset 1 (control).
        n2 : int
            Sample size of dataset 2 (treatment).
        unbiased : bool
            Return an unbiased estimate using Hedges' correction. Correction
            factor approximated using the formula proposed in Hedges 2011.
            (default = False)

        Returns
        -------
        smd : float
            Estimated standardized mean difference.
        se : float
            Standard error of the estimated standardized mean difference.
        """
        if (data1 and not data2) or (data2 and not data1):
            raise InputError('Two sets of data must be provided.')
        elif data1 and data2:
            if any([mean1, mean2, sd1, sd2, n1, n2]):
                warnings.warn("""Mean, n, and sd were computed from the data.
                                 These input args were ignored.""")
            mean1 = np.mean(data1)
            mean2 = np.mean(data2)
            sd1 = np.std(data1)
            sd2 = np.std(data2)
            n1 = len(data1)
            n2 = len(data2)

        # if (mean1 and not mean2) or (mean2 and not mean1):
        #     raise InputError('mean1 and mean2 must both be provided.')

        # if (sd1 and not sd2) or (sd2 and not sd1):
        #     raise InputError('sd1 and sd2 must both be provided.')

        # if (n1 and not n2) or (n2 and not n1):
        #     raise InputError('n1 and n2 must both be provided.')

        # cohens_d
        smd = (mean2 - mean1) / np.sqrt((sd1 ** 2 + sd2 ** 2) / 2)

        # standard error
        v_d = ((n1+n2) / (n1*n2)) + ((smd ** 2) / (2*(n1+n2)))
        se = np.sqrt(v_d)

        if unbiased:
            # Hedges correction (J. Hedges, 1981)
            # Approximation for the the correction factor from:
            # Introduction to Meta-Analysis. Michael Borenstein,
            # L. V. Hedges, J. P. T. Higgins and H. R. Rothstein
            # Wiley (2011). Chapter 4. Effect Sizes Based on Means.
            j = 1 - (3/(4*(n1+n2-2)-1))
            smd = j * smd
            v_g = (j ** 2) * v_d
            se = np.sqrt(v_g)

        return smd, se

    def _cat_smd(self, prop1=None, prop2=None, n1=None, n2=None,
                 unbiased=False):
        """
        Compute the standardized mean difference (regular or unbiased) using
        either raw data or summary measures.

        Parameters
        ----------
        prop1 : list
            Proportions (range 0-1) for each categorical value in dataset 1
            (control).
        prop2 : list
            Proportions (range 0-1) for each categorical value in dataset 2
            (treatment).
        n1 : int
            Sample size of dataset 1 (control).
        n2 : int
            Sample size of dataset 2 (treatment).
        unbiased : bool
            Return an unbiased estimate using Hedges' correction. Correction
            factor approximated using the formula proposed in Hedges 2011.
            (default = False)

        Returns
        -------
        smd : float
            Estimated standardized mean difference.
        se : float
            Standard error of the estimated standardized mean difference.
        """
        # Categorical SMD Yang & Dalton 2012
        # https://support.sas.com/resources/papers/proceedings12/335-2012.pdf
        prop1 = np.asarray(prop1)
        prop2 = np.asarray(prop2)

        # Drop first level for consistency with R tableone
        # "to eliminate dependence if more than two levels"
        prop1 = prop1[1:]
        prop2 = prop2[1:]

        lst_cov = []
        for p in [prop1, prop2]:
            variance = p * (1 - p)
            covariance = - np.outer(p, p)
            covariance[np.diag_indices_from(covariance)] = variance
            lst_cov.append(covariance)

        mean_diff = np.matrix(prop2 - prop1)
        mean_cov = (lst_cov[0] + lst_cov[1])/2

        # TODO: add steps to deal with nulls

        try:
            sq_md = mean_diff * np.linalg.inv(mean_cov) * mean_diff.T
        except LinAlgError:
            sq_md = np.nan

        try:
            smd = np.asarray(np.sqrt(sq_md))[0][0]
        except IndexError:
            smd = np.nan

        # standard error
        v_d = ((n1+n2) / (n1*n2)) + ((smd ** 2) / (2*(n1+n2)))
        se = np.sqrt(v_d)

        if unbiased:
            # Hedges correction (J. Hedges, 1981)
            # Approximation for the the correction factor from:
            # Introduction to Meta-Analysis. Michael Borenstein,
            # L. V. Hedges, J. P. T. Higgins and H. R. Rothstein
            # Wiley (2011). Chapter 4. Effect Sizes Based on Means.
            j = 1 - (3/(4*(n1+n2-2)-1))
            smd = j * smd
            v_g = (j ** 2) * v_d
            se = np.sqrt(v_g)

        return smd, se

    def _q25(self, x):
        """
        Compute percentile (25th)
        """
        return np.nanpercentile(x.values, 25)

    def _q75(self, x):
        """
        Compute percentile (75th)
        """
        return np.nanpercentile(x.values, 75)

    def _std(self, x):
        """
        Compute standard deviation with ddof degrees of freedom
        """
        if len(x) == 1:
            return 0.0
        else:
            return np.nanstd(x.values, ddof=self._ddof)

    def _diptest(self, x):
        """
        Compute Hartigan Dip Test for modality.

        p < 0.05 suggests possible multimodality.
        """
        p = hartigan_diptest(x.values)
        # dropna=False argument in pivot_table does not function as expected
        # https://github.com/pandas-dev/pandas/issues/22159
        # return -1 instead of None
        if pd.isnull(p):
            return -1
        return p

    def _normaltest(self, x):
        """
        Compute test for normal distribution.

        Null hypothesis: x comes from a normal distribution
        p < alpha suggests the null hypothesis can be rejected.
        """
        if len(x.values[~np.isnan(x.values)]) >= 20:
            stat, p = stats.normaltest(x.values, nan_policy='omit')
        else:
            p = None
        # dropna=False argument in pivot_table does not function as expected
        # return -1 instead of None
        if pd.isnull(p):
            return -1
        return p

    def _tukey(self, x, threshold):
        """
        Count outliers according to Tukey's rule.

        Where Q1 is the lower quartile and Q3 is the upper quartile,
        an outlier is an observation outside of the range:

        [Q1 - k(Q3 - Q1), Q3 + k(Q3 - Q1)]

        k = 1.5 indicates an outlier
        k = 3.0 indicates an outlier that is "far out"
        """
        vals = x.values[~np.isnan(x.values)]

        try:
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            low_bound = q1 - (iqr * threshold)
            high_bound = q3 + (iqr * threshold)
            outliers = np.where((vals > high_bound) | (vals < low_bound))
        except IndexError:
            outliers = []

        return outliers

    def _outliers(self, x):
        """
        Compute number of outliers
        """
        outliers = self._tukey(x, threshold=1.5)
        return np.size(outliers)

    def _far_outliers(self, x):
        """
        Compute number of "far out" outliers
        """
        outliers = self._tukey(x, threshold=3.0)
        return np.size(outliers)

    def _t1_summary(self, x):
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
                    np.nanmedian(x.values), np.nanmin(x.values),
                    np.nanmax(x.values),
                )
            else:
                return f.format(
                    np.nanmedian(x.values),
                    np.nanpercentile(x.values, 25),
                    np.nanpercentile(x.values, 75),
                )
        else:
            if self._min_max and x.name in self._min_max:
                f = "{{:.{}f}} [{{:.{}f}},{{:.{}f}}]".format(n, n, n)
                return f.format(
                    np.nanmean(x.values), np.nanmin(x.values),
                    np.nanmax(x.values),
                )
            else:
                f = '{{:.{}f}} ({{:.{}f}})'.format(n, n)
                return f.format(np.nanmean(x.values), self._std(x))

    def _create_cont_describe(self, data, groupby):
        """
        Describe the continuous data.

        Parameters
        ----------
            data : pandas DataFrame
                The input dataset.

        Returns
        ----------
            df_cont : pandas DataFrame
                Summarise the continuous variables.
        """
        aggfuncs = [pd.Series.count, np.mean, np.median, self._std,
                    self._q25, self._q75, min, max, self._t1_summary,
                    self._diptest, self._outliers, self._far_outliers,
                    self._normaltest]

        # coerce continuous data to numeric
        cont_data = data[self._continuous].apply(pd.to_numeric,
                                                 errors='coerce')
        # check all data in each continuous column is numeric
        bad_cols = cont_data.count() != data[self._continuous].count()
        bad_cols = cont_data.columns[bad_cols]
        if len(bad_cols) > 0:
            msg = ("The following continuous column(s) have "
                   "non-numeric values: {variables}. Either specify the "
                   "column(s) as categorical or remove the "
                   "non-numeric values.").format(variables=bad_cols.values)
            raise InputError(msg)

        # check for coerced column containing all NaN to warn user
        for column in cont_data.columns[cont_data.count() == 0]:
            self._non_continuous_warning(column)

        if groupby:
            # add the groupby column back
            cont_data = cont_data.merge(data[[groupby]],
                                        left_index=True,
                                        right_index=True)

            # group and aggregate data
            df_cont = pd.pivot_table(cont_data,
                                     columns=[groupby],
                                     aggfunc=aggfuncs)
        else:
            # if no groupby, just add single group column
            df_cont = cont_data.apply(aggfuncs).T
            df_cont.columns.name = 'Overall'
            df_cont.columns = pd.MultiIndex.from_product([df_cont.columns,
                                                         ['Overall']])

        df_cont.index = df_cont.index.rename('variable')

        # remove prefix underscore from column names (e.g. _std -> std)
        agg_rename = df_cont.columns.levels[0]
        agg_rename = [x[1:] if x[0] == '_' else x for x in agg_rename]
        df_cont.columns = df_cont.columns.set_levels(agg_rename, level=0)

        return df_cont

    def _format_cat(self, row):
        var = row.name[0]
        if var in self._decimals:
            n = self._decimals[var]
        else:
            n = 1
        f = '{{:.{}f}}'.format(n)
        return f.format(row.percent)

    def _create_cat_describe(self, data, groupby, groupbylvls):
        """
        Describe the categorical data.

        Parameters
        ----------
            data : pandas DataFrame
                The input dataset.

        Returns
        ----------
            df_cat : pandas DataFrame
                Summarise the categorical variables.
        """
        group_dict = {}

        for g in groupbylvls:
            if groupby:
                d_slice = data.loc[data[groupby] == g, self._categorical]
            else:
                d_slice = data[self._categorical].copy()

            # create a dataframe with freq, proportion
            df = d_slice.copy()

            # convert to str to handle int converted to boolean. Avoid nans.
            for column in df.columns:
                df[column] = [str(row) if not pd.isnull(row)
                              else None for row in df[column].values]

            df = df.melt().groupby(['variable',
                                    'value']).size().to_frame(name='freq')

            df['percent'] = df['freq'].div(df.freq.sum(level=0),
                                           level=0).astype(float) * 100

            # set number of decimal places for percent
            if isinstance(self._decimals, int):
                n = self._decimals
                f = '{{:.{}f}}'.format(n)
                df['percent_str'] = df['percent'].astype(float).map(f.format)
            elif isinstance(self._decimals, dict):
                df.loc[:, 'percent_str'] = df.apply(self._format_cat, axis=1)
            else:
                n = 1
                f = '{{:.{}f}}'.format(n)
                df['percent_str'] = df['percent'].astype(float).map(f.format)

            # add n column, listing total non-null values for each variable
            ct = d_slice.count().to_frame(name='n')
            ct.index.name = 'variable'
            df = df.join(ct)

            # add null count
            nulls = d_slice.isnull().sum().to_frame(name='Missing')
            nulls.index.name = 'variable'
            # only save null count to the first category for each variable
            # do this by extracting the first category from the df row index
            levels = df.reset_index()[['variable',
                                       'value']].groupby('variable').first()
            # add this category to the nulls table
            nulls = nulls.join(levels)
            nulls = nulls.set_index('value', append=True)
            # join nulls to categorical
            df = df.join(nulls)

            # add summary column
            df['t1_summary'] = (df.freq.map(str) + ' ('
                                + df.percent_str.map(str)+')')

            # add to dictionary
            group_dict[g] = df

        df_cat = pd.concat(group_dict, axis=1)
        # ensure the groups are the 2nd level of the column index
        if df_cat.columns.nlevels > 1:
            df_cat = df_cat.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)

        return df_cat

    def _create_htest_table(self, data):
        """
        Create a table containing P-Values for significance tests. Add features
        of the distributions and the P-Values to the dataframe.

        Parameters
        ----------
            data : pandas DataFrame
                The input dataset.

        Returns
        ----------
            df : pandas DataFrame
                A table containing the P-Values, test name, etc.
        """
        # list features of the variable e.g. matched, paired, n_expected
        df = pd.DataFrame(index=self._continuous+self._categorical,
                          columns=['continuous', 'nonnormal',
                                   'min_observed', 'P-Value', 'Test'])

        df.index = df.index.rename('variable')
        df['continuous'] = np.where(df.index.isin(self._continuous),
                                    True, False)

        df['nonnormal'] = np.where(df.index.isin(self._nonnormal),
                                   True, False)

        # list values for each variable, grouped by groupby levels
        for v in df.index:
            is_continuous = df.loc[v]['continuous']
            is_categorical = ~df.loc[v]['continuous']
            is_normal = ~df.loc[v]['nonnormal']

            # if continuous, group data into list of lists
            if is_continuous:
                catlevels = None
                grouped_data = {}
                for s in self._groupbylvls:
                    lvl_data = data.loc[data[self._groupby] == s, v]
                    # coerce to numeric and drop non-numeric data
                    lvl_data = lvl_data.apply(pd.to_numeric,
                                              errors='coerce').dropna()
                    # append to overall group data
                    grouped_data[s] = lvl_data.values
                min_observed = min([len(x) for x in grouped_data.values()])
            # if categorical, create contingency table
            elif is_categorical:
                catlevels = sorted(data[v].astype('category').cat.categories)
                cross_tab = pd.crosstab(data[self._groupby].
                                        rename('_groupby_var_'), data[v])
                min_observed = cross_tab.sum(axis=1).min()
                grouped_data = cross_tab.T.to_dict('list')

            # minimum number of observations across all levels
            df.loc[v, 'min_observed'] = min_observed

            # compute pvalues
            (df.loc[v, 'P-Value'],
                df.loc[v, 'Test']) = self._p_test(v, grouped_data,
                                                  is_continuous,
                                                  is_categorical, is_normal,
                                                  min_observed, catlevels)

        return df

    def _create_smd_table(self, data):
        """
        Create a table containing pairwise Standardized Mean Differences
        (SMDs).

        Parameters
        ----------
            data : pandas DataFrame
                The input dataset.

        Returns
        ----------
            df : pandas DataFrame
                A table containing pairwise standardized mean differences
                (SMDs).
        """
        # create the SMD table
        permutations = [sorted((x, y),
                        key=lambda f: self._groupbylvls.index(f))
                        for x in self._groupbylvls
                        for y in self._groupbylvls if x is not y]

        p_set = set(tuple(x) for x in permutations)

        colname = 'SMD ({0},{1})'
        columns = [colname.format(x[0], x[1]) for x in p_set]
        df = pd.DataFrame(index=self._continuous+self._categorical,
                          columns=columns)
        df.index = df.index.rename('variable')

        for p in p_set:
            try:
                for v in self.cont_describe.index:
                    smd, _ = self._cont_smd(
                                mean1=self.cont_describe['mean'][p[0]].loc[v],
                                mean2=self.cont_describe['mean'][p[1]].loc[v],
                                sd1=self.cont_describe['std'][p[0]].loc[v],
                                sd2=self.cont_describe['std'][p[1]].loc[v],
                                n1=self.cont_describe['count'][p[0]].loc[v],
                                n2=self.cont_describe['count'][p[1]].loc[v],
                                unbiased=False)
                    df[colname.format(p[0], p[1])].loc[v] = smd
            except AttributeError:
                pass

            try:
                for v, _ in self.cat_describe.groupby(level=0):
                    smd, _ = self._cat_smd(
                        prop1=self.cat_describe.loc[[v]]['percent'][p[0]].values/100,
                        prop2=self.cat_describe.loc[[v]]['percent'][p[1]].values/100,
                        n1=self.cat_describe.loc[[v]]['freq'][p[0]].sum(),
                        n2=self.cat_describe.loc[[v]]['freq'][p[1]].sum(),
                        unbiased=False)
                    df[colname.format(p[0], p[1])].loc[v] = smd
            except AttributeError:
                pass

        return df

    def _p_test(self, v, grouped_data, is_continuous, is_categorical,
                is_normal, min_observed, catlevels):
        """
        Compute P-Values.

        Parameters
        ----------
            v : str
                Name of the variable to be tested.
            grouped_data : dict
                Dictionary of Numpy Arrays to be tested.
            is_continuous : bool
                True if the variable is continuous.
            is_categorical : bool
                True if the variable is categorical.
            is_normal : bool
                True if the variable is normally distributed.
            min_observed : int
                Minimum number of values across groups for the variable.
            catlevels : list
                Sorted list of levels for categorical variables.

        Returns
        ----------
            pval : float
                The computed P-Value.
            ptest : str
                The name of the test used to compute the P-Value.
        """

        # no test by default
        pval = np.nan
        ptest = 'Not tested'

        # apply user defined test
        if self._htest and v in self._htest:
            pval = self._htest[v](*grouped_data.values())
            ptest = self._htest[v].__name__
            return pval, ptest

        # do not test if the variable has no observations in a level
        if min_observed == 0:
            msg = ("No P-Value was computed for {variable} due to the low "
                   "number of observations.""").format(variable=v)
            warnings.warn(msg)
            return pval, ptest

        # continuous
        if (is_continuous and is_normal and len(grouped_data) == 2
            and min_observed >= 2):
            ptest = 'Two Sample T-test'
            test_stat, pval = stats.ttest_ind(*grouped_data.values(),
                                              equal_var=False,
                                              nan_policy="omit")
        elif is_continuous and is_normal:
            # normally distributed
            ptest = 'One-way ANOVA'
            test_stat, pval = stats.f_oneway(*grouped_data.values())
        elif is_continuous and not is_normal:
            # non-normally distributed
            ptest = 'Kruskal-Wallis'
            test_stat, pval = stats.kruskal(*grouped_data.values())
        # categorical
        elif is_categorical:
            # default to chi-squared
            ptest = 'Chi-squared'
            grouped_val_list = [x for x in grouped_data.values()]
            chi2, pval, dof, expected = stats.chi2_contingency(grouped_val_list)
            # if any expected cell counts are < 5, chi2 may not be valid
            # if this is a 2x2, switch to fisher exact
            if expected.min() < 5 or min_observed < 5:
                if np.shape(grouped_val_list) == (2, 2):
                    ptest = "Fisher's exact"
                    odds_ratio, pval = stats.fisher_exact(grouped_val_list)
                else:
                    ptest = "Chi-squared (warning: expected count < 5)"
                    chi_warn = ("Chi-squared tests for the following "
                                "variables may be invalid due to the low "
                                "number of observations")
                    try:
                        self._warnings[chi_warn].append(v)
                    except KeyError:
                        self._warnings[chi_warn] = [v]

        return pval, ptest

    def _create_cont_table(self, data, overall):
        """
        Create tableone for continuous data.

        Returns
        ----------
        table : pandas DataFrame
            A table summarising the continuous variables.
        """
        # remove the t1_summary level
        table = self.cont_describe[['t1_summary']].copy()
        table.columns = table.columns.droplevel(level=0)

        # add a column of null counts as 1-count() from previous function
        nulltable = data[self._continuous].isnull().sum().to_frame(name='Missing')
        try:
            table = table.join(nulltable)
        # if columns form a CategoricalIndex, need to convert to string first
        except TypeError:
            table.columns = table.columns.astype(str)
            table = table.join(nulltable)

        # add an empty value column, for joining with cat table
        table['value'] = ''
        table = table.set_index([table.index, 'value'])

        # add pval column
        if self._pval and self._pval_adjust:
            table = table.join(self._htest_table[['P-Value (adjusted)',
                                                  'Test']])
        elif self._pval:
            table = table.join(self._htest_table[['P-Value', 'Test']])

        # add standardized mean difference (SMD) column/s
        if self._smd:
            table = table.join(self.smd_table)

        # join the overall column if needed
        if self._groupby and overall:
            table = table.join(pd.concat([self.cont_describe_all['t1_summary'].
                                          Overall], axis=1, keys=["Overall"]))

        return table

    def _create_cat_table(self, data, overall):
        """
        Create table one for categorical data.

        Returns
        ----------
        table : pandas DataFrame
            A table summarising the categorical variables.
        """
        table = self.cat_describe['t1_summary'].copy()

        # add the total count of null values across all levels
        isnull = data[self._categorical].isnull().sum().to_frame(name='Missing')
        isnull.index = isnull.index.rename('variable')
        try:
            table = table.join(isnull)
        # if columns form a CategoricalIndex, need to convert to string first
        except TypeError:
            table.columns = table.columns.astype(str)
            table = table.join(isnull)

        # add pval column
        if self._pval and self._pval_adjust:
            table = table.join(self._htest_table[['P-Value (adjusted)',
                                                  'Test']])
        elif self._pval:
            table = table.join(self._htest_table[['P-Value', 'Test']])

        # add standardized mean difference (SMD) column/s
        if self._smd:
            table = table.join(self.smd_table)

        # join the overall column if needed
        if self._groupby and overall:
            table = table.join(pd.concat([self.cat_describe_all['t1_summary'].
                                          Overall], axis=1, keys=["Overall"]))

        return table

    def _create_tableone(self, data):
        """
        Create table 1 by combining the continuous and categorical tables.

        Returns
        ----------
        table : pandas DataFrame
            The complete table one.
        """
        if self._continuous and self._categorical:
            # support pandas<=0.22
            try:
                table = pd.concat([self.cont_table, self.cat_table],
                                  sort=False)
            except TypeError:
                table = pd.concat([self.cont_table, self.cat_table])
        elif self._continuous:
            table = self.cont_table
        elif self._categorical:
            table = self.cat_table

        # ensure column headers are strings before reindexing
        table = table.reset_index().set_index(['variable', 'value'])
        table.columns = table.columns.values.astype(str)

        # sort the table rows
        sort_columns = ['Missing', 'P-Value', 'P-Value (adjusted)', 'Test']
        if self._smd:
            sort_columns = sort_columns + list(self.smd_table.columns)

        if self._sort and isinstance(self._sort, bool):
            new_index = sorted(table.index.values, key=lambda x: x[0].lower())
        elif self._sort and isinstance(self._sort, str) and (self._sort in
                                                             sort_columns):
            try:
                new_index = table.sort_values(self._sort).index
            except KeyError:
                new_index = sorted(table.index.values,
                                   key=lambda x: self._columns.index(x[0]))
                warnings.warn('Sort variable not found: {}'.format(self._sort))
        elif self._sort and isinstance(self._sort, str) and (self._sort not in
                                                             sort_columns):
            new_index = sorted(table.index.values,
                               key=lambda x: self._columns.index(x[0]))
            warnings.warn('Sort must be in the following ' +
                          'list: {}.'.format(self._sort))
        else:
            # sort by the columns argument
            new_index = sorted(table.index.values,
                               key=lambda x: self._columns.index(x[0]))
        table = table.reindex(new_index)

        # round pval column and convert to string
        if self._pval and self._pval_adjust:
            table['P-Value (adjusted)'] = table['P-Value (adjusted)'].apply(
                                                '{:.3f}'.format).astype(str)
            table.loc[table['P-Value (adjusted)'] == '0.000',
                      'P-Value (adjusted)'] = '<0.001'
        elif self._pval:
            table['P-Value'] = table['P-Value'].apply(
                                     '{:.3f}'.format).astype(str)
            table.loc[table['P-Value'] == '0.000', 'P-Value'] = '<0.001'

        # round smd columns and convert to string
        if self._smd:
            for c in list(self.smd_table.columns):
                table[c] = table[c].apply('{:.3f}'.format).astype(str)
                table.loc[table[c] == '0.000', c] = '<0.001'

        # if an order is specified, apply it
        if self._order:
            for k in self._order:

                # Skip if the variable isn't present
                try:
                    all_var = table.loc[k].index.unique(level='value')
                except KeyError:
                    if k not in self._groupby:
                        warnings.warn("Order variable not found: {}".format(k))
                    continue

                # Remove value from order if it is not present
                if [i for i in self._order[k] if i not in all_var]:
                    rm_var = [i for i in self._order[k] if i not in all_var]
                    self._order[k] = [i for i in self._order[k]
                                      if i in all_var]
                    warnings.warn(("Order value not found: "
                                   "{}: {}").format(k, rm_var))

                new_seq = [(k, '{}'.format(v)) for v in self._order[k]]
                new_seq += [(k, '{}'.format(v)) for v in all_var
                            if v not in self._order[k]]

                # restructure to match the original idx
                new_idx_array = np.empty((len(new_seq),), dtype=object)
                new_idx_array[:] = [tuple(i) for i in new_seq]
                orig_idx = table.index.values.copy()
                orig_idx[table.index.get_loc(k)] = new_idx_array
                table = table.reindex(orig_idx)

        # set the limit on the number of categorical variables
        if self._limit:
            levelcounts = data[self._categorical].nunique()
            for k, _ in levelcounts.iteritems():

                # set the limit for the variable
                if (isinstance(self._limit, int)
                        and levelcounts[k] >= self._limit):
                    limit = self._limit
                elif isinstance(self._limit, dict) and k in self._limit:
                    limit = self._limit[k]
                else:
                    continue

                if not self._order or (self._order and k not in self._order):
                    # re-order the variables by frequency
                    count = data[k].value_counts().sort_values(ascending=False)
                    new_idx = [(k, '{}'.format(i)) for i in count.index]
                else:
                    # apply order
                    all_var = table.loc[k].index.unique(level='value')
                    new_idx = [(k, '{}'.format(v)) for v in self._order[k]]
                    new_idx += [(k, '{}'.format(v)) for v in all_var
                                if v not in self._order[k]]

                # restructure to match the original idx
                new_idx_array = np.empty((len(new_idx),), dtype=object)
                new_idx_array[:] = [tuple(i) for i in new_idx]
                orig_idx = table.index.values.copy()
                orig_idx[table.index.get_loc(k)] = new_idx_array
                table = table.reindex(orig_idx)

                # drop the rows > the limit
                table = table.drop(new_idx_array[limit:])

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

        # only display data in first level row
        dupe_mask = table.groupby(level=[0]).cumcount().ne(0)
        dupe_columns = ['Missing']
        optional_columns = ['P-Value', 'P-Value (adjusted)', 'Test']
        if self._smd:
            optional_columns = optional_columns + list(self.smd_table.columns)
        for col in optional_columns:
            if col in table.columns.values:
                dupe_columns.append(col)

        table[dupe_columns] = table[dupe_columns].mask(dupe_mask).fillna('')

        # remove Missing column if not needed
        if not self._isnull:
            table = table.drop('Missing', axis=1)

        if self._pval and not self._pval_test_name:
            table = table.drop('Test', axis=1)

        # replace nans with empty strings
        table = table.fillna('')

        # add column index
        if not self._groupbylvls == ['Overall']:
            # rename groupby variable if requested
            c = self._groupby
            if self._alt_labels:
                if self._groupby in self._alt_labels:
                    c = self._alt_labels[self._groupby]

            c = 'Grouped by {}'.format(c)
            table.columns = pd.MultiIndex.from_product([[c], table.columns])

        # display alternative labels if assigned
        table = table.rename(index=self._create_row_labels(), level=0)

        # ensure the order of columns is consistent
        if self._groupby and self._order and (self._groupby in self._order):
            header = ['{}'.format(v) for v in table.columns.levels[1].values]
            cols = self._order[self._groupby] + ['{}'.format(v)
                                                 for v in header
                                                 if v not in
                                                 self._order[self._groupby]]
        elif self._groupby:
            cols = ['{}'.format(v) for v in table.columns.levels[1].values]
        else:
            cols = ['{}'.format(v) for v in table.columns.values]

        if self._groupby and self._overall:
            cols = ['Overall'] + [x for x in cols if x != 'Overall']

        if 'Missing' in cols:
            cols = ['Missing'] + [x for x in cols if x != 'Missing']

        # move optional_columns to the end of the dataframe
        for col in optional_columns:
            if col in cols:
                cols = [x for x in cols if x != col] + [col]

        if self._groupby:
            table = table.reindex(cols, axis=1, level=1)
        else:
            table = table.reindex(cols, axis=1)

        try:
            if 'Missing' in self._alt_labels or 'Overall' in self._alt_labels:
                table = table.rename(columns=self._alt_labels)
        except TypeError:
            pass

        # remove the 'variable, value' column names in the index
        table = table.rename_axis([None, None])

        return table

    def _create_row_labels(self):
        """
        Take the original labels for rows. Rename if alternative labels are
        provided. Append label suffix if label_suffix is True.

        Returns
        ----------
        labels : dictionary
            Dictionary, keys are original column name, values are final label.

        """
        # start with the original column names
        labels = {}
        for c in self._columns:
            labels[c] = c

        # replace column names with alternative names if provided
        if self._alt_labels:
            for k in self._alt_labels.keys():
                labels[k] = self._alt_labels[k]

        # append the label suffix
        if self._label_suffix:
            for k in labels.keys():
                if k in self._nonnormal:
                    if self._min_max and k in self._min_max:
                        labels[k] = "{}, {}".format(labels[k], "median [min,max]")
                    else:
                        labels[k] = "{}, {}".format(labels[k], "median [Q1,Q3]")
                elif k in self._categorical:
                    labels[k] = "{}, {}".format(labels[k], "n (%)")
                else:
                    if self._min_max and k in self._min_max:
                        labels[k] = "{}, {}".format(labels[k], "mean [min,max]")
                    else:
                        labels[k] = "{}, {}".format(labels[k], "mean (SD)")

        return labels

    # warnings
    def _non_continuous_warning(self, c):
        msg = ("'{}' has all non-numeric values. Consider including "
               "it in the list of categorical variables.").format(c)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)


# Allow TableOne to be called as a function.
# Refactor this out at some point!
@docstring_copier(TableOne.__doc__)
def tableone(*args, **kwargs):
    """{0}"""
    return TableOne(*args, **kwargs)
