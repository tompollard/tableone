from typing import Optional

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

    def format_cat(self, row, col, decimals) -> str:
        """
        Format values to n decimal places.
        """
        var = row.name[0]
        if var in decimals:
            n = decimals[var]  # type: ignore
        else:
            n = 1
        f = '{{:.{}f}}'.format(n)

        return f.format(row[col])

    def create_cat_describe(self,
                            data: pd.DataFrame,
                            categorical,
                            decimals,
                            row_percent,
                            groupby: Optional[str] = None,
                            groupbylvls: Optional[list] = None
                            ) -> pd.DataFrame:
        """
        Describe the categorical data.

        Parameters
        ----------
            data : pandas DataFrame
                The input dataset.
            groupby : Str
                Variable to group by.
            groupbylvls : List
                List of levels in the groupby variable.

        Returns
        ----------
            df_cat : pandas DataFrame
                Summarise the categorical variables.
        """
        group_dict = {}

        cat_slice = data[categorical].copy()

        for g in groupbylvls:  # type: ignore
            if groupby:
                df = cat_slice.loc[data[groupby] == g, categorical]
            else:
                df = cat_slice.copy()

            # create n column and null count column
            # must be done before converting values to strings
            ct = df.count().to_frame(name='n')
            ct.index.name = 'variable'
            nulls = df.isnull().sum().to_frame(name='Missing')
            nulls.index.name = 'variable'

            # Convert to str to handle int converted to boolean in the index.
            # Also avoid nans.
            for column in df.columns:
                df[column] = [str(row) if not pd.isnull(row)
                              else None for row in df[column].values]
                cat_slice[column] = [str(row) if not pd.isnull(row)
                                     else None for row
                                     in cat_slice[column].values]

            # create a dataframe with freq, proportion
            df = df.melt().groupby(['variable',
                                    'value']).size().to_frame(name='freq')

            df['percent'] = df['freq'].div(df.groupby(level=0).freq.sum(),
                                           level=0).astype(float) * 100

            # add row percent
            df['percent_row'] = df['freq'].div(cat_slice[categorical]
                                               .melt()
                                               .groupby(['variable', 'value'])
                                               .size()) * 100

            # set number of decimal places for percent
            if isinstance(decimals, int):
                n = decimals
                f = '{{:.{}f}}'.format(n)
                df['percent_str'] = df['percent'].astype(float).map(f.format)
                df['percent_row_str'] = df['percent_row'].astype(float).map(
                    f.format)
            elif isinstance(decimals, dict):
                df.loc[:, 'percent_str'] = df.apply(self.format_cat, axis=1,
                                                    args=['percent', decimals])
                df.loc[:, 'percent_row_str'] = df.apply(self.format_cat, axis=1,
                                                        args=['percent_row', decimals])
            else:
                n = 1
                f = '{{:.{}f}}'.format(n)
                df['percent_str'] = df['percent'].astype(float).map(f.format)
                df['percent_row_str'] = df['percent_row'].astype(float).map(
                    f.format)

            # join count column
            df = df.join(ct)

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
            if row_percent:
                df['t1_summary'] = (df.freq.map(str) + ' ('
                                    + df.percent_row_str.map(str)+')')
            else:
                df['t1_summary'] = (df.freq.map(str) + ' ('
                                    + df.percent_str.map(str)+')')

            # add to dictionary
            group_dict[g] = df

        df_cat = pd.concat(group_dict, axis=1)
        # ensure the groups are the 2nd level of the column index
        if df_cat.columns.nlevels > 1:
            df_cat = df_cat.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)

        return df_cat
