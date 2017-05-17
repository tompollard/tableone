"""
Package for producing Table 1 in medical research papers, 
inspired by the R package of the same name.
"""

__author__ = "Tom Pollard <tpollard@mit.edu>"
__version__ = "0.1.2"

import pandas as pd 
from tabulate import tabulate


class TableOne(object):

    def __init__(self, data, numerical=None, categorical=None, strata_col=None):

        # instance variables
        # self.data = data
        self.numerical = numerical
        self.categorical = categorical
        self.strata_col = strata_col
        self._cont_describe = {}
        self._cat_describe = {}

        if strata_col:
            self.strata = data[strata_col][data[strata_col].notnull()].astype('category').unique().categories.sort_values()
        else:
            self.strata = ['overall']
        
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
        self.tableone = self.__create_tableone(data)

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
        cont_describe = pd.DataFrame(index=self.numerical)
        cont_describe['n'] = data[self.numerical].count().values
        cont_describe['isnull'] = data[self.numerical].isnull().sum().values
        cont_describe['mean'] = data[self.numerical].mean().values
        cont_describe['std'] = data[self.numerical].std().values
        cont_describe['q25'] = data[self.numerical].quantile(0.25).values
        cont_describe['q75'] = data[self.numerical].quantile(0.75).values
        cont_describe['min'] = data[self.numerical].min().values
        cont_describe['max'] = data[self.numerical].max().values
        cont_describe['skew'] = data[self.numerical].skew().values
        cont_describe['kurt'] = data[self.numerical].kurt().values

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

        for v in self.numerical:
            row = ['{} (mean (std))'.format(v)]
            for strata in self._cont_describe.iterkeys():
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

    def __create_tableone(self,data):
        """
        Create table 1 by combining the continuous and categorical tables
        """
        header = [] 

        if not self.strata_col:
            # s_row = ['']
            # s_row.append("{}".format('overall'))
            # header.append(s_row)
            n_row = ['n'] 
            n_row.append("{}".format(len(data.index)))
            header.append(n_row)
        elif self.strata_col:
            pass

        table = header + self._cont_table + self._cat_table
        
        return table    

    def to_csv(self):
        pass

