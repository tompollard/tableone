import warnings

import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
from scipy import stats
from statsmodels.stats import multitest

from tableone.modality import hartigan_diptest
from tableone.exceptions import InputError


class Statistics:
    def __init__(self):
        """
        Initialize the Statistics class, which provides statistical methods used by TableOne.
        """
        pass

    def _q25(self, x):
        """
        Compute 25th percentile
        """
        return np.nanpercentile(x.values, 25)

    def _q75(self, x):
        """
        Compute 75th percentile
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
        Compute Hartigan Dip Test for modality (test the hypothesis that the data is unimodal).

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
        Compute the number of outliers based on Tukey's test with a threshold of 1.5.
        """
        outliers = self._tukey(x, threshold=1.5)
        return np.size(outliers)

    def _far_outliers(self, x) -> int:
        """
        Compute the number of "far out" outliers based on Tukey's test with a threshold of 3.0.
        """
        outliers = self._tukey(x, threshold=3.0)
        return np.size(outliers)

    def _normality(self, x):
        """
        Perform a test for normality.

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
        smd = (mean2 - mean1) / np.sqrt((sd1 ** 2 + sd2 ** 2) / 2)  # type: ignore

        # standard error
        v_d = ((n1+n2) / (n1*n2)) + ((smd ** 2) / (2*(n1+n2)))  # type: ignore
        se = np.sqrt(v_d)

        if unbiased:
            # Hedges correction (J. Hedges, 1981)
            # Approximation for the the correction factor from:
            # Introduction to Meta-Analysis. Michael Borenstein,
            # L. V. Hedges, J. P. T. Higgins and H. R. Rothstein
            # Wiley (2011). Chapter 4. Effect Sizes Based on Means.
            j = 1 - (3/(4*(n1+n2-2)-1))  # type: ignore
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
            covariance = - np.outer(p, p)  # type: ignore
            covariance[np.diag_indices_from(covariance)] = variance
            lst_cov.append(covariance)

        mean_diff = np.asarray(prop2 - prop1).reshape((1, -1))  # type: ignore
        mean_cov = (lst_cov[0] + lst_cov[1])/2

        # TODO: add steps to deal with nulls

        try:
            sq_md = mean_diff @ np.linalg.inv(mean_cov) @ mean_diff.T
        except LinAlgError:
            sq_md = np.nan

        try:
            smd = np.asarray(np.sqrt(sq_md))[0][0]
        except IndexError:
            smd = np.nan

        # standard error
        v_d = ((n1+n2) / (n1*n2)) + ((smd ** 2) / (2*(n1+n2)))  # type: ignore
        se = np.sqrt(v_d)

        if unbiased:
            # Hedges correction (J. Hedges, 1981)
            # Approximation for the the correction factor from:
            # Introduction to Meta-Analysis. Michael Borenstein,
            # L. V. Hedges, J. P. T. Higgins and H. R. Rothstein
            # Wiley (2011). Chapter 4. Effect Sizes Based on Means.
            j = 1 - (3/(4*(n1+n2-2)-1))  # type: ignore
            smd = j * smd
            v_g = (j ** 2) * v_d
            se = np.sqrt(v_g)

        return smd, se
