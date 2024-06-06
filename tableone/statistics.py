import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats import multitest

from tableone.modality import hartigan_diptest


class Statistics:
    def __init__(self):
        """Initialize the Statistics class."""
        pass

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

    def _std(self, x, ddof):
        """
        Compute standard deviation with ddof degrees of freedom
        """
        if len(x) == 1:
            return 0.0
        else:
            return np.nanstd(x.values, ddof=ddof)

    def _tukey(self, x, threshold):
        """
        Find outliers according to Tukey's rule.

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

    def _hartigan_dip(self, x):
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

    def _outliers(self, x) -> int:
        """
        Compute number of outliers
        """
        outliers = self._tukey(x, threshold=1.5)
        return np.size(outliers)

    def _far_outliers(self, x) -> int:
        """
        Compute number of "far out" outliers
        """
        outliers = self._tukey(x, threshold=3.0)
        return np.size(outliers)

    def _normality(self, x):
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

    def _p_test(self, v: str,
                grouped_data: dict,
                is_continuous: bool,
                is_categorical: bool,
                is_normal: bool,
                min_observed: int,
                catlevels: list,
                h_test: dict):
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
        warning_msg = None

        # apply user defined test
        if h_test and v in h_test:
            pval = h_test[v](*grouped_data.values())
            ptest = h_test[v].__name__
            return pval, ptest, warning_msg

        # do not test if the variable has no observations in a level
        if min_observed == 0:
            msg = ("No P-Value was computed for {variable} due to the low "
                   "number of observations.""").format(variable=v)
            warnings.warn(msg)
            return pval, ptest, warning_msg

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
            _, pval, _, expected = stats.chi2_contingency(
                grouped_val_list)
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
                    warning_msg = chi_warn

        return pval, ptest, warning_msg

    def multipletests(self, pvals, alpha, method):
        """
        Apply correction to p values for multiple testing.
        """
        return multitest.multipletests(pvals, alpha=alpha, method=method)
