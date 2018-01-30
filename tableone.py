"""
Package for producing Table 1 in medical research papers,
inspired by the R package of the same name.
"""

__author__ = "Tom Pollard <tpollard@mit.edu>, Alistair Johnson"
__version__ = "0.4.1"

import pandas as pd
import csv
from scipy import stats
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
        ddof (int): Degrees of freedom for standard deviation calculations (default: 1). 
        labels (Dict): Dictionary of alternative labels for variables. e.g. {'sex':'gender', 'trt':'treatment'}.
        limit (Int): Limit to the top N most frequent categories.
    """

    def __init__(self, data, columns='autodetect', categorical='autodetect', 
        groupby='', nonnormal=[], pval=False, isnull=True, ddof=1, labels=None, limit=None):

        # check input arguments
        if groupby and type(groupby) == list:
            groupby = groupby[0]
        if nonnormal and type(nonnormal) == str:
            nonnormal = [nonnormal]

        # if columns not specified, use all columns
        if columns == 'autodetect':
            columns = data.columns.get_values()

        # if categorical not specified, try to identify categorical
        if categorical == 'autodetect':
            categorical = self._detect_categorical_columns(data[columns])

        if pval and not groupby:
            raise ValueError("If pval=True then the groupby must be specified.")

        self.columns = columns
        self.isnull = isnull
        self.continuous = [c for c in columns if c not in categorical + [groupby]]
        self.categorical = categorical
        self.nonnormal = nonnormal
        self.pval = pval
        # self.sort = sort
        self.groupby = groupby
        self.ddof = ddof # degrees of freedom for standard deviation calculations
        self.labels = labels
        self.limit = limit

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
        # assume all non-numerical and date columns are categorical
        numeric_cols = set(data._get_numeric_data().columns.values)
        date_cols = set(data.select_dtypes(include=[np.datetime64]).columns)
        likely_cat = set(data.columns) - numeric_cols
        likely_cat = list(likely_cat - date_cols)
        # check proportion of unique values if numerical
        for var in data._get_numeric_data().columns:
            likely_flag = 1.0 * data[var].nunique()/data[var].count() < 0.05
            if likely_flag:
                 likely_cat.append(var)
        print(likely_cat)
        return likely_cat

    def q25(self,x):
        return np.nanpercentile(x.values,25)

    def q75(self,x):
        return np.nanpercentile(x.values,75)

    def std(self,x):
        return np.nanstd(x.values,ddof=self.ddof)

    def t1_summary(self,x):
        if x.name in self.nonnormal:
            return '{:.2f} [{:.2f},{:.2f}]'.format(np.nanmedian(x.values), 
                np.nanpercentile(x.values,25), np.nanpercentile(x.values,75))
        else:
            return '{:.2f} ({:.2f})'.format(np.nanmean(x.values), 
                np.nanstd(x.values,ddof=self.ddof))

    def _create_cont_describe(self,data):
        """
        Describe the continuous data.
        """
        aggfuncs = [pd.Series.count,np.mean,np.median,self.std,
            self.q25,self.q75,min,max,self.t1_summary]

        if self.groupby:
            cont_data = data[self.continuous + [self.groupby]]
            cont_data = cont_data.apply(pd.to_numeric, errors='ignore')
            df_cont = pd.pivot_table(cont_data,
                columns=[self.groupby],
                aggfunc=aggfuncs)
        else:
            # if no groupby, just add single group column
            df_cont = data[self.continuous].apply(pd.to_numeric, 
                errors='ignore').apply(aggfuncs).T
            df_cont.columns.name = 'overall'
            df_cont.columns = pd.MultiIndex.from_product([df_cont.columns, 
                ['overall']])

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

        # do not test if the variable has no observations in a level
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
        Create a table displaying tableone for continuous data.
        """
        # remove the t1_summary level
        table = self._cont_describe[['t1_summary']].copy()
        table.columns = table.columns.droplevel(level=0)

        # add a column of null counts
        nulltable = pd.DataFrame(data[self.continuous].isnull().sum().rename('isnull'))
        table = table.join(nulltable)

        # add an empty level column, for joining with cat table
        table['level'] = ''
        table.set_index([table.index,'level'],inplace=True)

        # add pval column
        if self.pval:
            table = table.join(self._significance_table[['pval','ptest']])

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

        # round pval column
        if self.pval:
            table['pval'] = table['pval'].apply('{:.3f}'.format)

        # sort the table
        table.reset_index().set_index(['variable','level'], inplace=True)
        table.sort_index(level='variable', inplace=True, sort_remaining=False)

        # if a limit has been set on the number of categorical variables
        # then order the variables by frequency
        if self.limit:
            levelcounts = data[self.categorical].nunique()
            levelcounts = levelcounts[levelcounts >= self.limit]
            for v,_ in levelcounts.iteritems():
                count = data[v].value_counts().sort_values(ascending=False)
                new_index = [(v, i) for i in count.index]
                old_index = table.index.values.copy()
                old_index[table.index.get_loc(v)] = new_index
                table = table.reindex(old_index)

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

        # remove empty column added above
        table.drop([''], axis=1, inplace=True)

        # remove isnull column if not needed
        if not self.isnull:
            table.drop('isnull',axis=1,inplace=True)

        # replace nans with empty strings
        table.fillna('',inplace=True)

        # add name of groupby variable to column headers
        if not self.groupbylvls == ['overall']:
            table.rename(columns=lambda x: x if x not in self.groupbylvls \
                else '{}={}'.format(self.groupby,x), inplace=True)

        # display alternative labels if assigned
        if self.labels:
            table.rename(index=self.labels, inplace=True, level=0)

        # if a limit has been set on the number of categorical variables
        # limit the number of categorical variables that are displayed
        if self.limit:
            table = table.groupby('variable').head(self.limit)

        return table
