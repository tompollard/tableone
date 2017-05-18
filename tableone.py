"""
Package for producing Table 1 in medical research papers,
inspired by the R package of the same name.
"""

__author__ = "Tom Pollard <tpollard@mit.edu>"
__version__ = "0.1.10"

import pandas as pd
from tabulate import tabulate
import csv


class TableOne(object):
    """
    Create a tableone instance.

    Args:
        data (Pandas DataFrame): The dataset to be summarised.
        continuous (List): List of column names for the continuous variables.
        categorical (List): List of column names for the categorical variables.
        strata_col (String): Column name for stratification (default None).
        nonnormal (List): List of column names for non-normal variables (default None).
    """

    def __init__(self, data, continuous=None, categorical=None, strata_col=None, nonnormal=[]):

        # instance variables
        self.continuous = continuous
        self.categorical = categorical
        self.strata_col = strata_col
        self.nonnormal = nonnormal
        self._cont_describe = {}
        self._cat_describe = {}

        if strata_col:
            self.strata = data[strata_col][data[strata_col].notnull()].astype('category').unique().categories.sort_values()
        else:
            self.strata = ['overall']

        self._n_row = self.__create_n_row(data)

        for s in self.strata:
            if strata_col:
                self._cont_describe[s] = self.__create_cont_describe(data.loc[data[strata_col] == s])
                self._cat_describe[s] = self.__create_cat_describe(data.loc[data[strata_col] == s])
            else:
                self._cont_describe[s] = self.__create_cont_describe(data)
                self._cat_describe[s] = self.__create_cat_describe(data)

        # create tables of continuous and categorical variables
        self._cont_table = self.__create_cont_table()
        self._cat_table = self.__create_cat_table(data)

        # combine continuous variables and categorical variables into table 1
        self.tableone = self.__create_tableone()

    def __str__(self):
        return self.__pretty_print_table()

    def __repr__(self):
        return self.__pretty_print_table()

    def __pretty_print_table(self):
        if self.strata_col:
            strat_str = 'Stratified by ' + '{}\n'.format(self.strata_col)
        else:
            strat_str = 'Overall\n'
        headers = [''] + sorted(self._cat_describe.keys())
        table = tabulate(self.tableone, headers = headers)
        return strat_str + table

    def __create_cont_describe(self,data):
        """
        Describe the continuous data
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

    def __create_cat_describe(self,data):
        """
        Describe the categorical data
        """
        cats = {}

        for v in self.categorical:
            ds = data[v].astype('category')
            df = pd.DataFrame(index=range(len(ds[ds.notnull()].cat.codes.unique())))
            df['n'] = ds.count()
            df['isnull'] = ds.isnull().sum()
            df['level'] = ds[ds.notnull()].unique().categories.sort_values()
            df = df.merge(ds.value_counts(dropna=True).to_frame().rename(columns= {v:'freq'}),
                left_on='level',right_index=True)
            df['percent'] = (df['freq'] / df['n']) * 100
            cats[v] = df

        return cats

    def __create_cont_table(self):
        """
        Create a table displaying table one for continuous data
        """
        table = []

        for v in self.continuous:
            if v in self.nonnormal:
                row = ['{} (median [IQR])'.format(v)]
            else:
                row = ['{} (mean (std))'.format(v)]
            for strata in self._cont_describe.iterkeys():
                if v in self.nonnormal:
                    row.append("{:0.2f} [{:0.2f},{:0.2f}]".format(self._cont_describe[strata]['median'][v],
                        self._cont_describe[strata]['q25'][v],
                        self._cont_describe[strata]['q75'][v]))
                else:
                    row.append("{:0.2f} ({:0.2f})".format(self._cont_describe[strata]['mean'][v],
                        self._cont_describe[strata]['std'][v]))                    
            # stack rows to create the table
            table.append(row)

        return table

    def __create_cat_table(self,data):
        """
        Create a table displaying table one for categorical data
        """
        table = []

        # For each variable
        # oh dear the loops
        for v in self.categorical:
            row = ['{} (n (%))'.format(v)]
            row.append('')
            table.append(row)
            # For each level within the variable
            for level in data[v][data[v].notnull()].astype('category').unique().categories.sort_values():
                row = ["{}".format(level)]
                # for each strata
                for strata in sorted(self._cat_describe.iterkeys()):
                    vals = self._cat_describe[strata][v][self._cat_describe[strata][v]['level']==level]
                    freq = vals['freq'].values[0]
                    percent = vals['percent'].values[0]
                    row.append("{:0.2f} ({:0.2f})".format(freq,percent))
                table.append(row)

        return table

    def __create_n_row(self,data):
        """
        Get n, the number of rows for each strata
        """
        n = ['n']
        if self.strata_col:
            for s in self.strata:
                count = data[self.strata_col][data[self.strata_col]==s].count()
                n.append("{}".format(count))
        else:
            count = len(data.index)
            n.append("{}".format(count))

        return n

    def __create_tableone(self):
        """
        Create table 1 by combining the continuous and categorical tables
        """
        table = [self._n_row] + self._cont_table + self._cat_table

        return table

    def to_csv(self,fn='tableone.csv'):
        """
        Write tableone to CSV

        Args:
            fn (String): Filename (default 'tableone.csv')
        """
        with open(fn, "wb") as f:
            writer = csv.writer(f)
            writer.writerows(self.tableone)

    def to_html(self,fn='tableone.html'):
        """
        Write tableone to HTML
        Args:
            fn (String): Filename (default 'tableone.html')
        """
        tablefmt = 'html'
        with open(fn, "wb") as f:
            f.write(tabulate(self.tableone, tablefmt=tablefmt))

    def to_markdown(self,fn='tableone.md'):
        """
        Write tableone to markdown
        Args:
            fn (String): Filename (default 'tableone.md')
        """
        tablefmt = 'pipe'
        with open(fn, "wb") as f:
            f.write(tabulate(self.tableone, tablefmt=tablefmt))

    def to_latex(self,fn='tableone.tex'):
        """
        Write tableone to LaTeX
        Args:
            fn (String): Filename (default 'tableone.tex')
        """
        tablefmt = 'latex'
        with open(fn, "wb") as f:
            f.write(tabulate(self.tableone, tablefmt=tablefmt))
