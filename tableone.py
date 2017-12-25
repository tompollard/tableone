"""
Package for producing Table 1 in medical research papers,
inspired by the R package of the same name.
"""

__author__ = "Tom Pollard <tpollard@mit.edu>, Alistair Johnson"
__version__ = "0.3.1"

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
        sort (Boolean): Order the rows alphabetically, with exception to the 'n' row

    """

    def __init__(self, data, columns='autodetect', categorical='autodetect', 
        groupby='', nonnormal=[], pval=False, isnull=True, sort=True):

        # check input arguments
        if groupby and type(groupby) == list:
            groupby = groupby[0]
        if nonnormal and type(nonnormal) == str:
            nonnormal = [nonnormal]

        # if categorical not specified
        # try to identify categorical
        if categorical == 'autodetect':
            categorical = self._detect_categorical_columns(data)

        # if columns not specified
        # use all columns
        if columns == 'autodetect':
            columns = data.columns.get_values()

        if pval and not groupby:
            raise ValueError("If pval=True then the groupby must be specified.")

        self.columns = columns
        self.isnull = isnull
        self.continuous = [c for c in columns if c not in categorical + [groupby]]
        self.categorical = categorical
        self.nonnormal = nonnormal
        self.pval = pval
        self.sort = sort
        self.groupby = groupby

        if self.groupby:
            self.groupbylvls = sorted(data.groupby(groupby).groups.keys())
        else:
            self.groupbylvls = ['overall']

        # forgive me jraffa
        if self.pval:
            self._significance_table = self._create_significance_table(data)

        # create descriptive tables
        if self.categorical:
            self._cat_describe = self._create_cat_describe(data)
            self._cat_table = self._create_cat_table(data)

        # create tables of continuous and categorical variables
        if self.continuous:
            self._cont_describe = self._create_cont_describe(data)
            self._cont_table = self._create_cont_table(data)

        # combine continuous variables and categorical variables into table 1
        self.tableone = self._create_tableone(data)

    def __str__(self):
        return self.tableone.to_string()

    def __repr__(self):
        return self.tableone.to_string() 

    def _detect_categorical_columns(self,data):
        """
        If categorical is not specified, auto-detect categorical columns
        """
        likely_cat = []
        for var in data.columns:
            likely_flag = 1.0 * data[var].nunique()/data[var].count() < 0.05
            if likely_flag:
                 likely_cat.append(var)

        return likely_cat

    def _create_cont_describe(self,data):
        """
        Describe the continuous data.
        """
        group_dict = {}

        for g in self.groupbylvls:
            if self.groupby:
                d_slice = data.loc[data[self.groupby] == g]
            else: 
                d_slice = data
            df = pd.DataFrame(index=self.continuous)
            df.index.name = 'variable'
            df = pd.concat([df,d_slice[self.continuous].count()],axis=1)
            df = pd.concat([df,data[self.continuous].isnull().sum().rename('isnull')],axis=1)
            df = pd.concat([df,d_slice[self.continuous].mean().rename('mean')],axis=1)
            df = pd.concat([df,d_slice[self.continuous].median().rename('median')],axis=1)
            df = pd.concat([df,d_slice[self.continuous].std().rename('std')],axis=1)
            df = pd.concat([df,d_slice[self.continuous].quantile(0.25).rename('q25')],axis=1)
            df = pd.concat([df,d_slice[self.continuous].quantile(0.75).rename('q75')],axis=1)
            df = pd.concat([df,d_slice[self.continuous].min().rename('min')],axis=1)
            df = pd.concat([df,d_slice[self.continuous].max().rename('max')],axis=1)
            df['isnormal'] = np.where(~df.index.isin(self.nonnormal),1,0)
            df['t1_summary_txt'] = np.where(df['isnormal'] == 1,'(mean (std))','(median [IQR])')
            df['iqr'] = '[' + df['q25'].apply(round,ndigits=2).map(str) + ', ' + df['q75'].apply(round,ndigits=2).map(str) + ']'
            df['t1_summary'] = np.where(df['isnormal'] == 1,
                df['mean'].apply(round,ndigits=2).map(str) + ' (' + df['std'].apply(round,ndigits=2).map(str) + ')',
                df['median'].apply(round,ndigits=2).map(str) + ' ' + df['iqr'])
            group_dict[g] = df

        df_cont = pd.concat(group_dict,axis=1)
        df_cont.index.rename('variable',inplace=True)

        return df_cont

    def _create_cat_describe(self,data):
        """
        Describe the categorical data.
        """
        group_dict = {}

        for g in self.groupbylvls:
            if self.groupby:
                d_slice = data.loc[data[self.groupby] == g]
            else: 
                d_slice = data
            cats = {}

            for v in self.categorical:
                ds = d_slice[v].astype('category')
                levels = ds[ds.notnull()].unique().categories.sort_values()
                df = pd.DataFrame(index = levels)
                # clean later
                # add descriptive details
                df['n'] = ds.count()
                df['isnull'] = data[v].isnull().sum()
                df['level'] = levels
                df = df.merge(ds.value_counts(dropna=True).to_frame().rename(columns= {v:'freq'}),
                    left_on='level',right_index=True, how='left')
                df['freq'].fillna(0,inplace=True)
                df['percent'] = (df['freq'] / df['n']) * 100
                # set level as index to df
                df.set_index('level', inplace=True)
                cats[v] = df

            cats_df = pd.concat(cats)
            cats_df.index.rename('variable',level=0, inplace=True)

            cats_df['t1_summary'] = cats_df.freq.map(str) \
                + ' (' + cats_df.percent.apply(round, ndigits=2).map(str) + ')'

            group_dict[g] = cats_df

        df_cat = pd.concat(group_dict,axis=1)

        return df_cat

    def _create_significance_table(self,data):
        """
        Create a table containing p values for significance tests. Add features of
        the distributions and the p values to the dataframe.
        """

        # list features of the variable e.g. matched, paired, n_expected
        df=pd.DataFrame(index=self.continuous+self.categorical,
            columns=['continuous','nonnormal','min_observed','pval','ptest'])

        df.index.rename('variable', inplace=True)
        df['continuous'] = np.where(df.index.isin(self.continuous),True,False)
        df['nonnormal'] = np.where(df.index.isin(self.nonnormal),True,False)

        # list values for each variable, grouped by groupby levels
        for v in df.index:
            
            # compute p value
            is_continuous = df.loc[v]['continuous']
            is_categorical = ~df.loc[v]['continuous']
            is_normal = ~df.loc[v]['nonnormal']

            # if continuous, group data into list of lists
            if is_continuous:
                catlevels = None
                grouped_data = []
                for s in self.groupbylvls:
                    lvl_data = data[data[self.groupby]==s].dropna(subset=[v])[v]
                    grouped_data.append(lvl_data.values)
                min_observed = len(min(grouped_data,key=len))
            # if categorical, create contingency table
            elif is_categorical:
                catlevels = sorted(data[v].astype('category').cat.categories)
                grouped_data = pd.crosstab(data[self.groupby],data[v])
                min_observed = grouped_data.sum(axis=1).min()

            # minimum number of observations across all levels
            df.loc[v,'min_observed'] = min_observed

            # compute pvalues
            df.loc[v,'pval'],df.loc[v,'ptest'] = self._p_test(v, 
                grouped_data,is_continuous,is_categorical,
                is_normal,min_observed,catlevels)

        return df

    def _p_test(self,v,grouped_data,is_continuous,is_categorical,
            is_normal,min_observed,catlevels,
            pval=np.nan,ptest='Not tested'):
        """
        Compute p value
        """

        # do not test if any sub-group has no observations
        if min_observed == 0:
            warnings.warn('No p-value was computed for {} due to the low number of observations.'.format(v))
            return pval,ptest

        # continuous
        if is_continuous and is_normal:
            # normally distributed
            ptest = 'One-way ANOVA'
            test_stat, pval = stats.f_oneway(*grouped_data)
        elif is_continuous and not is_normal:
            # non-normally distributed
            ptest = 'Kruskal-Wallis'
            test_stat, pval = stats.kruskal(*grouped_data)
        # categorical
        elif is_categorical:
            # default to chi-squared
            ptest = 'Chi-squared'
            chi2, pval, dof, expected = stats.chi2_contingency(grouped_data)
            # if any expected cell counts are < 5, chi2 may not be valid
            # if this is a 2x2, switch to fisher exact
            if expected.min() < 5:
                if grouped_data.shape == (2,2):
                    ptest = 'Fisher''s exact'
                    oddsratio, pval = stats.fisher_exact(grouped_data)
                else:
                    ptest = 'Chi-squared (warning: expected count < 5)'
                    warnings.warn('No p-value was computed for {} due to the low number of observations.'.format(v))

        return pval,ptest

    def _create_cont_table(self,data):
        """
        Create a table displaying table one for continuous data.
        """
        table = self._cont_describe[self.groupbylvls[0]][['isnull']].copy()
        
        for g in self.groupbylvls:
            table[g] = self._cont_describe[g]['t1_summary']

        table['level'] = ''
        table.set_index([table.index,'level'],inplace=True)

        # add pval column
        if self.pval:
            table = table.join(self._significance_table[['pval','ptest']])
            # table['pval'] = table['pval'].round(3)

        return table

    def _create_cat_table(self,data):
        """
        Create a table displaying table one for categorical data.
        """
        table = self._cat_describe[self.groupbylvls[0]][['isnull']].copy()
        
        for g in self.groupbylvls:
            table[g] = self._cat_describe[g]['t1_summary']

        # add pval column
        if self.pval:
            table = table.join(self._significance_table[['pval','ptest']])

        return table

    def _create_tableone(self,data):
        """
        Create table 1 by combining the continuous and categorical tables.
        """
        if self.continuous and self.categorical:
            table = pd.concat([self._cont_table,self._cat_table])
        elif self.continuous:
            table = self._cont_table
        elif self.categorical:
            table = self._cat_table

        if self.sort:
            table.sort_index(level='variable', inplace=True)

        # round pval column
        if self.pval:
            table['pval'] = table['pval'].apply('{:.3f}'.format)

        # inserts n row
        n_row = pd.DataFrame(columns = ['variable','level','isnull'])
        n_row.set_index(['variable','level'], inplace=True)
        n_row.loc['n', ''] = None
        table = pd.concat([n_row,table])

        if self.groupbylvls == ['overall']:
            table.loc['n','overall'] = len(data.index)
        else:
            for g in self.groupbylvls:
                ct = data[self.groupby][data[self.groupby]==g].count()
                table.loc['n',g] = ct

        # only display data in first level row
        dupe_mask = table.groupby(level=[0]).cumcount().ne(0)
        dupe_columns = ['isnull']
        if ['pval'] in table.columns.values:
            dupe_columns.append('pval')
        if ['ptest'] in table.columns.values:
            dupe_columns.append('ptest')         
        table[dupe_columns] = table[dupe_columns].mask(dupe_mask).fillna('')

        # remove isnull column if not needed
        if not self.isnull:
            table.drop('isnull',axis=1,inplace=True)

        # replace nans with empty strings
        table.fillna('',inplace=True)

        # add name of groupby variable to column headers
        if not self.groupbylvls == ['overall']:
            table.rename(columns=lambda x: x if x not in self.groupbylvls \
                else '{}={}'.format(self.groupby,x), inplace=True)

        return table
