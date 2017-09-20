"""
Package for producing Table 1 in medical research papers,
inspired by the R package of the same name.
"""

__author__ = "Tom Pollard <tpollard@mit.edu>, Alistair Johnson"
__version__ = "0.2.8"

import pandas as pd
from tabulate import tabulate
import csv
from scipy import stats
from collections import Counter, OrderedDict
import warnings
import numpy as np


class TableOne(object):
    """
    Create a tableone instance.

    Args:
        data (Pandas DataFrame): The dataset to be summarised. Rows are observations, columns are variables.
        columns (List): List of columns in the dataset to be included in the final table.
        categorical (List): List of columns that contain categorical variables.
        groupby (String): Optional column for stratifying the final table (default None).
        nonnormal (List): List of columns that contain non-normal variables (default None).
        pval (Boolean): Whether to display computed P values (default False).
        isnull (Boolean): Whether to display a count of null values (default True).

    """

    def __init__(self, data, columns=[], categorical=[], groupby='', nonnormal=[], pval=False, isnull=True):

        # check input arguments
        if groupby and type(groupby) == list:
            groupby = groupby[0]
        if nonnormal and type(nonnormal) == str:
            nonnormal = [nonnormal]

        if pval and not groupby:
            raise ValueError("If pval=True then the groupby must be specified.")

        # instance variables
        self.columns = columns
        self.isnull = isnull
        self.continuous = [c for c in columns if c not in categorical + [groupby]]
        self.categorical = categorical
        self.groupby = groupby
        self.nonnormal = nonnormal
        self.pval = pval

        if groupby:
            self.strata = sorted(data.groupby(groupby).groups.keys())
        else:
            self.strata = ['overall']

        # forgive me jraffa
        if groupby:
            self._significance_table = self.__create_significance_table(data)

        self._n_row = self.__create_n_row(data)

        self._cont_describe = {}
        self._cat_describe = {}
        self._cat_levels = self.__get_cat_levels(data)

        for s in self.strata:
            if groupby:
                self._cont_describe[s] = self.__create_cont_describe(data.loc[data[groupby] == s])
                self._cat_describe[s] = self.__create_cat_describe(data.loc[data[groupby] == s])
            else:
                self._cont_describe[s] = self.__create_cont_describe(data)
                self._cat_describe[s] = self.__create_cat_describe(data)

        # create tables of continuous and categorical variables
        self._cont_table = self.__create_cont_table(data)
        self._cat_table = self.__create_cat_table(data)

        # combine continuous variables and categorical variables into table 1
        self.tableone = self.__create_tableone()

    def __str__(self):
        return self.__pretty_print_table()

    def __repr__(self):
        return self.__pretty_print_table()

    def __pretty_print_table(self):
        """
        Print formatted table to screen.
        """
        if self.groupby:
            strat_str = 'Stratified by ' + '{}\n'.format(self.groupby)
        else:
            strat_str = 'Overall\n'
        headers = [''] + self.strata

        if self.isnull:
            headers.append('isnull')

        if self.pval:
            headers.append('pval')
            headers.append('testname')

        table = tabulate(self.tableone, headers = headers)

        return strat_str + table

    def __create_cont_describe(self,data):
        """
        Describe the continuous data.
        """
        if self.continuous:
            cont_describe = pd.DataFrame(index=self.continuous)
            cont_describe['n'] = data[self.continuous].count().values
            cont_describe['isnull'] = data[self.continuous].isnull().sum().values
            cont_describe['mean'] = data[self.continuous].mean().values
            cont_describe['median'] = data[self.continuous].median().values
            cont_describe['std'] = data[self.continuous].std().values
            cont_describe['q25'] = data[self.continuous].quantile(0.25).values
            cont_describe['q75'] = data[self.continuous].quantile(0.75).values
            cont_describe['min'] = data[self.continuous].min().values
            cont_describe['max'] = data[self.continuous].max().values
            cont_describe['skew'] = data[self.continuous].skew().values
            cont_describe['kurt'] = data[self.continuous].kurt().values
        else:
            cont_describe = []

        return cont_describe

    def __get_cat_levels(self,data):
        """
        Get a full list of levels for each categorical variable
        """
        levels = {}

        for v in self.categorical:
            ds = data[v].astype('category')
            levels[v] = ds[ds.notnull()].unique().categories.sort_values()

        return levels

    def __create_cat_describe(self,data):
        """
        Describe the categorical data.
        """
        cats = {}

        for v in self.categorical:
            ds = data[v].astype('category')
            df = pd.DataFrame(index=range(len(self._cat_levels[v])))
            df['n'] = ds.count()
            df['isnull'] = ds.isnull().sum()
            df['level'] = self._cat_levels[v]
            df = df.merge(ds.value_counts(dropna=True).to_frame().rename(columns= {v:'freq'}),
                left_on='level',right_index=True, how='left')
            df['freq'].fillna(0,inplace=True)
            df['percent'] = (df['freq'] / df['n']) * 100
            # set level as index to df
            df.set_index('level', inplace=True)
            cats[v] = df

        return cats

    def __create_significance_table(self,data):
        """
        Create a table containing p values for significance tests. Add features of
        the distributions and the p values to the dataframe.
        """

        # list features of the variable e.g. matched, paired, n_expected
        df=pd.DataFrame(index=self.continuous+self.categorical,
            columns=['continuous','nonnormal','min_n','pval','testname'])

        for v in self.continuous + self.categorical:
            # is the variable continuous?
            if v in self.categorical:
                df.loc[v]['continuous'] = 0
            else:
                df.loc[v]['continuous'] = 1
            # is the variable reported to be nonnormal?
            if v in self.nonnormal:
                df.loc[v]['nonnormal'] = 1
            else:
                df.loc[v]['nonnormal'] = 0
            # group the data for analysis
            grouped_data = []
            for s in self.strata:
                grouped_data.append(data[v][data[self.groupby]==s][data[v][data[self.groupby]==s].notnull()].values)
            # minimum n across groups
            df.loc[v]['min_n'] = len(min(grouped_data,key=len))
            if self.pval:
                # compute p value
                df.loc[v]['pval'],df.loc[v]['testname'] = self.__p_test(df,v,grouped_data,data)

        return df

    def __p_test(self,df,v,grouped_data,data):
        """
        Compute p value
        """

        # default, don't test
        pval = np.nan
        testname = 'Not tested'

        # do not test if any sub-group has no observations
        if df.loc[v]['min_n'] == 0:
            warnings.warn('No p-value was computed for {} due to the low number of observations.'.format(v))
            return pval,testname

        # continuous
        if df.loc[v]['continuous'] == 1:
            if df.loc[v]['nonnormal'] == 0:
                # normally distributed
                testname = 'One_way_ANOVA'
                test_stat, pval = stats.f_oneway(*grouped_data)
            elif df.loc[v]['nonnormal'] == 1:
                # non-normally distributed
                testname = 'Kruskal-Wallis'
                test_stat, pval = stats.kruskal(*grouped_data)
        # categorical
        elif df.loc[v]['continuous'] == 0:
            # get the ordered observed frequencies of each level within each strata
            all_lvls = sorted(data[v][data[v].notnull()].unique())
            grp_counts = [dict(Counter(g)) for g in grouped_data]
            # make sure that all_lvls are represented in the grp_counts
            for d in grp_counts:
                for k in all_lvls:
                    if k not in d:
                        d[k] = 0

            # now make sure that the ordered dictionaries have the same order
            # getting messy, clean up
            grp_counts_ordered = list()
            for d in grp_counts:
                d_ordered = OrderedDict()
                for k in all_lvls:
                    d_ordered[k] = d[k]
                grp_counts_ordered.append(d_ordered)

            observed = [list(g.values()) for g in grp_counts_ordered]

            # if any of the cell counts are < 5, we shouldn't use chi2
            if min((min(x) for x in observed)) < 5:
                # switch to fisher exact if this is a 2x2
                if (len(observed)==2) & (len(observed[0])==2):
                    testname = 'Fisher exact'
                    oddsratio, pval = stats.fisher_exact(observed)
                else:
                    warnings.warn('No p-value was computed for {} due to the low number of observations.'.format(v))
                    # otherwise, we will not test
            else:
                testname = 'Chi-squared'
                chi2, pval, dof, expected = stats.chi2_contingency(observed)

        return pval,testname

    def __create_cont_table(self,data):
        """
        Create a table displaying table one for continuous data.
        """
        table = []

        for v in self.continuous:
            if v in self.nonnormal:
                row = ['{} (median [IQR])'.format(v)]
            else:
                row = ['{} (mean (std))'.format(v)]
            for strata in self.strata:
                if v in self.nonnormal:
                    row.append("{:0.2f} [{:0.2f},{:0.2f}]".format(self._cont_describe[strata]['median'][v],
                        self._cont_describe[strata]['q25'][v],
                        self._cont_describe[strata]['q75'][v]))
                else:
                    row.append("{:0.2f} ({:0.2f})".format(self._cont_describe[strata]['mean'][v],
                        self._cont_describe[strata]['std'][v]))

            # add isnull values column
            if self.isnull:
                row.append(data[v].isnull().sum())

            # add pval column
            if self.pval:
                row.append('{:0.3f}'.format(self._significance_table.loc[v].pval))
                row.append('{}'.format(self._significance_table.loc[v].testname))

            # stack rows to create the table
            table.append(row)

        return table

    def __create_cat_table(self,data):
        """
        Create a table displaying table one for categorical data.
        """
        table = []

        # For each variable
        # oh dear the loops
        for v in self.categorical:
            row = ['{} (n (%))'.format(v)]
            row = row + len(self.strata) * ['']

            # add isnull values column
            if self.isnull:
                row.append(data[v].isnull().sum())

            # add pval column
            if self.pval:
                row.append('{:0.3f}'.format(self._significance_table.loc[v].pval))
                row.append('{}'.format(self._significance_table.loc[v].testname))

            table.append(row)

            # For each level within the variable
            for level in data[v][data[v].notnull()].astype('category').unique().categories.sort_values():
                row = ["{}".format(level)]
                # for each strata
                for strata in self.strata:
                    # get data frame with info about each individual level
                    vals = self._cat_describe[strata][v]
                    freq = vals.loc[level, 'freq']
                    percent = vals.loc[level, 'percent']

                    row.append("{:0.0f} ({:0.2f})".format(freq,percent))
                # stack rows to create the table
                table.append(row)

        return table

    def __create_n_row(self,data):
        """
        Get n, the number of rows for each strata.
        """
        n = ['n']
        if self.groupby:
            for s in self.strata:
                count = data[self.groupby][data[self.groupby]==s].count()
                n.append('{:0.0f}'.format(count))
            if self.isnull:
                isnull = data[self.groupby].isnull().sum()
                n.append('{:0.0f}'.format(isnull))
        else:
            count = len(data.index)
            n.append("{:0.0f}".format(count))
            if self.isnull:
                n.append('')

        if self.pval:
            n.append('')

        return n

    def __create_tableone(self):
        """
        Create table 1 by combining the continuous and categorical tables.
        """
        table = [self._n_row] + self._cont_table + self._cat_table

        return table

    def to_csv(self,fn='tableone.csv'):
        """
        Write tableone to CSV.

        Args:
            fn (String): Filename (default 'tableone.csv')
        """
        with open(fn, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(self.tableone)

    def to_html(self,fn='tableone.html'):
        """
        Write tableone to HTML.

        Args:
            fn (String): Filename (default 'tableone.html')
        """
        tablefmt = 'html'
        with open(fn, 'w') as f:
            f.write(tabulate(self.tableone, tablefmt=tablefmt))

    def to_markdown(self,fn='tableone.md'):
        """
        Write tableone to markdown.

        Args:
            fn (String): Filename (default 'tableone.md')
        """
        tablefmt = 'pipe'
        with open(fn, 'w') as f:
            f.write(tabulate(self.tableone, tablefmt=tablefmt))

    def to_latex(self,fn='tableone.tex'):
        """
        Write tableone to LaTeX.

        Args:
            fn (String): Filename (default 'tableone.tex')
        """
        tablefmt = 'latex'
        with open(fn, 'w') as f:
            f.write(tabulate(self.tableone, tablefmt=tablefmt))
