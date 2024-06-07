import numpy as np
import pandas as pd

from tableone.statistics import Statistics


class Tables:
    """
    Create and store intermediate tables used to create Table 1.

    Usage:

    self.tables = Tables()
    self.tables._create_htest_table()
    self.tables._htest_table
    """
    def __init__(self):
        """
        Initialize the Tables class.
        """
        self.statistics = Statistics()

    def create_htest_table(self, data: pd.DataFrame,
                           continuous,
                           categorical,
                           nonnormal,
                           groupby,
                           groupbylvls,
                           htest,
                           pval,
                           pval_adjust) -> pd.DataFrame:
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
        df = pd.DataFrame(index=continuous+categorical,
                          columns=['continuous', 'nonnormal',
                                   'min_observed', 'P-Value', 'Test'])

        df.index = df.index.rename('variable')

        df['continuous'] = np.where(df.index.isin(continuous), True, False)
        df['nonnormal'] = np.where(df.index.isin(nonnormal), True, False)

        # list values for each variable, grouped by groupby levels
        for v in df.index:
            is_continuous = df.loc[v]['continuous']
            is_categorical = ~df.loc[v]['continuous']
            is_normal = ~df.loc[v]['nonnormal']

            # if continuous, group data into list of lists
            if is_continuous:
                catlevels = None
                grouped_data = {}
                for s in groupbylvls:
                    lvl_data = data.loc[data[groupby] == s, v]
                    # coerce to numeric and drop non-numeric data
                    lvl_data = lvl_data.apply(pd.to_numeric,
                                              errors='coerce').dropna()
                    # append to overall group data
                    grouped_data[s] = lvl_data.values
                min_observed = min([len(x) for x in grouped_data.values()])
            # if categorical, create contingency table
            elif is_categorical:
                catlevels = sorted(data[v].astype('category').cat.categories)
                cross_tab = pd.crosstab(data[groupby].rename('_groupby_var_'), data[v])
                min_observed = cross_tab.sum(axis=1).min()
                grouped_data = cross_tab.T.to_dict('list')

            # minimum number of observations across all levels
            df.loc[v, 'min_observed'] = min_observed  # type: ignore

            # compute pvalues
            warning_msg = None
            (df.loc[v, 'P-Value'],
             df.loc[v, 'Test'],
             warning_msg) = self.statistics._p_test(v, grouped_data, is_continuous, is_categorical,  # type: ignore
                                                    is_normal,  min_observed, catlevels, htest)  # type: ignore

            # TODO: Improve method for handling these warnings.
            # Write to logfile?
            #
            # if warning_msg:
            #     try:
            #         self._warnings[warning_msg].append(v)
            #     except KeyError:
            #         self._warnings[warning_msg] = [v]

        # correct for multiple testing
        if pval and pval_adjust:
            adjusted = self.statistics.multipletests(df['P-Value'],
                                                     alpha=0.05,
                                                     method=pval_adjust)
            df['P-Value (adjusted)'] = adjusted[1]
            df['adjust method'] = pval_adjust

        return df

    def create_smd_table(self,
                         data: pd.DataFrame,
                         groupbylvls,
                         continuous,
                         categorical,
                         cont_describe,
                         cat_describe) -> pd.DataFrame:
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
                        key=lambda f: groupbylvls.index(f))
                        for x in groupbylvls
                        for y in groupbylvls if x is not y]

        p_set = set(tuple(x) for x in permutations)

        colname = 'SMD ({0},{1})'
        columns = [colname.format(x[0], x[1]) for x in p_set]
        df = pd.DataFrame(index=continuous+categorical, columns=columns)
        df.index = df.index.rename('variable')

        for p in p_set:
            try:
                for v in cont_describe.index:
                    smd, _ = self.statistics._cont_smd(
                                mean1=cont_describe['mean'][p[0]].loc[v],
                                mean2=cont_describe['mean'][p[1]].loc[v],
                                sd1=cont_describe['std'][p[0]].loc[v],
                                sd2=cont_describe['std'][p[1]].loc[v],
                                n1=cont_describe['count'][p[0]].loc[v],
                                n2=cont_describe['count'][p[1]].loc[v],
                                unbiased=False)
                    df.loc[v, colname.format(p[0], p[1])] = smd
            except AttributeError:
                pass

            try:
                for v, _ in cat_describe.groupby(level=0):
                    smd, _ = self.statistics._cat_smd(
                        prop1=cat_describe.loc[[v]]['percent'][p[0]].values/100,
                        prop2=cat_describe.loc[[v]]['percent'][p[1]].values/100,
                        n1=cat_describe.loc[[v]]['freq'][p[0]].sum(),
                        n2=cat_describe.loc[[v]]['freq'][p[1]].sum(),
                        unbiased=False)
                    df.loc[v, colname.format(p[0], p[1])] = smd  # type: ignore
            except AttributeError:
                pass

        return df
