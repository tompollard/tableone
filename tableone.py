"""
The tableone package simplifies producing a "Table 1" frequently used to summarize data in publications.
It provides the TableOne class, which can be called on a pandas dataframe.
This class contains a number of utilities for summarizing the data using commonly applied statistical measures.
"""

__author__ = "Tom Pollard <tpollard@mit.edu>, Alistair Johnson"
__version__ = "0.5.12"

import pandas as pd
from scipy import stats
import warnings
import numpy as np
from statsmodels.stats import multitest
import modality

class InputError(Exception):
    """
    Exception raised for errors in the input.
    """
    pass


class TableOne(object):
    """
    Create a tableone instance.

    Parameters
    ----------
    data : pandas DataFrame
        The dataset to be summarised. Rows are observations, columns are variables.
    columns : list, optional
        List of columns in the dataset to be included in the final table.
    categorical : list, optional
        List of columns that contain categorical variables.
    groupby : str, optional
        Optional column for stratifying the final table (default: None).
    nonnormal : list, optional
        List of columns that contain non-normal variables (default: None).
    pval : bool, optional
        Display computed p-values (default: False).
    pval_adjust : str, optional
        Method used to adjust p-values for multiple testing. 
        For a complete list, see documentation for statsmodels multipletests.
        Available methods include ::

        `None` : no correction applied.
        `bonferroni` : one-step correction
        `sidak` : one-step correction
        `holm-sidak` : step down method using Sidak adjustments
        `simes-hochberg` : step-up method (independent)
        `hommel` : closed method based on Simes tests (non-negative)

    isnull : bool, optional
        Display a count of null values (default: True).
    ddof : int, optional
        Degrees of freedom for standard deviation calculations (default: 1).
    labels : dict, optional
        Dictionary of alternative labels for variables.
        e.g. `labels = {'sex':'gender', 'trt':'treatment'}`
    sort : bool, optional
        Sort the rows alphabetically. Default (False) retains the input order
        of columns.
    limit : int, optional
        Limit to the top N most frequent categories.
    remarks : bool, optional
        Add remarks on the appropriateness of the summary measures and the
        statistical tests (default: True).

    Attributes
    ----------
    tableone : dataframe
        Summary of the data (i.e., the "Table 1").
    """

    def __init__(self, data, columns=None, categorical=None, groupby=None,
        nonnormal=None, pval=False, pval_adjust=None, isnull=True,
        ddof=1, labels=None, sort=False, limit=None, remarks=True):

        # check input arguments
        if not groupby:
            groupby = ''
        elif groupby and type(groupby) == list:
            groupby = groupby[0]

        if not nonnormal:
            nonnormal=[]
        elif nonnormal and type(nonnormal) == str:
            nonnormal = [nonnormal]

        # if columns not specified, use all columns
        if not columns:
            columns = data.columns.get_values()

        # check that the columns exist in the dataframe
        if not set(columns).issubset(data.columns):
            notfound = list(set(columns) - set(data.columns))
            raise InputError('Columns not found in dataset: {}'.format(notfound))

        # check for duplicate columns
        dups = data[columns].columns[data[columns].columns.duplicated()].unique()
        if not dups.empty:
            raise InputError('Input contains duplicate columns: {}'.format(dups))

        # if categorical not specified, try to identify categorical
        if not categorical and type(categorical) != list:
            categorical = self._detect_categorical_columns(data[columns])

        if pval and not groupby:
            raise InputError("If pval=True then the groupby must be specified.")

        self._columns = list(columns)
        self._isnull = isnull
        self._continuous = [c for c in columns if c not in categorical + [groupby]]
        self._categorical = categorical
        self._nonnormal = nonnormal
        self._pval = pval
        self._pval_adjust = pval_adjust
        self._sort = sort
        self._groupby = groupby
        self._ddof = ddof # degrees of freedom for standard deviation
        self._labels = labels
        self._limit = limit
        self._remarks = remarks

        # output column names that cannot be contained in a groupby
        self._reserved_columns = ['isnull', 'pval', 'ptest', 'pval (adjusted)']
        if self._groupby:
            self._groupbylvls = sorted(data.groupby(groupby).groups.keys())
            # check that the group levels do not include reserved words
            for level in self._groupbylvls:
                if level in self._reserved_columns:
                    raise InputError('Group level contained "{}", a reserved keyword for tableone.'.format(level))
        else:
            self._groupbylvls = ['overall']

        # forgive me jraffa
        if self._pval:
            self._significance_table = self._create_significance_table(data)

        # correct for multiple testing
        if self._pval and self._pval_adjust:
            alpha=0.05
            adjusted = multitest.multipletests(self._significance_table['pval'],
                alpha=alpha, method=self._pval_adjust)
            self._significance_table['pval (adjusted)'] = adjusted[1]
            self._significance_table['adjust method'] = self._pval_adjust

        # create descriptive tables
        if self._categorical:
            self.cat_describe = self._create_cat_describe(data)
            self.cat_table = self._create_cat_table(data)

        # create tables of continuous and categorical variables
        if self._continuous:
            self.cont_describe = self._create_cont_describe(data)
            self.cont_table = self._create_cont_table(data)

        # combine continuous variables and categorical variables into table 1
        self.tableone = self._create_tableone(data)
        # self._remarks_str = self._generate_remark_str()

        # wrap dataframe methods
        self.head = self.tableone.head
        self.tail = self.tableone.tail
        self.to_csv = self.tableone.to_csv
        self.to_excel = self.tableone.to_excel
        self.to_html = self.tableone.to_html
        self.to_json = self.tableone.to_json
        self.to_latex = self.tableone.to_latex

    def __str__(self):
        return self.tableone.to_string() + self._generate_remark_str('\n')

    def __repr__(self):
        return self.tableone.to_string() + self._generate_remark_str('\n')

    def _repr_html_(self):
        return self.tableone._repr_html_() + self._generate_remark_str('<br />')

    def _generate_remark_str(self, end_of_line = '\n'):
        """
        Generate a series of remarks that the user should consider
        when interpreting the summary statistics.
        """
        warnings = {}
        msg = '{}'.format(end_of_line)

        # generate warnings for continuous variables
        if self._continuous:
            # highlight far outliers
            outlier_mask = self.cont_describe.far_outliers > 1
            outlier_vars = list(self.cont_describe.far_outliers[outlier_mask].dropna(how='all').index)
            if outlier_vars:
                warnings["Warning, Tukey test indicates far outliers in"] = outlier_vars

            # highlight possible multimodal distributions using hartigan's dip test
            # -1 values indicate NaN
            modal_mask = (self.cont_describe.diptest >= 0) & (self.cont_describe.diptest <= 0.05)
            modal_vars = list(self.cont_describe.diptest[modal_mask].dropna(how='all').index)
            if modal_vars:
                warnings["Warning, Hartigan's Dip Test reports possible multimodal distributions for"] = modal_vars

            # highlight non normal distributions
            # -1 values indicate NaN
            modal_mask = (self.cont_describe.normaltest >= 0) & (self.cont_describe.normaltest <= 0.001)
            modal_vars = list(self.cont_describe.normaltest[modal_mask].dropna(how='all').index)
            if modal_vars:
                warnings["Warning, test for normality reports non-normal distributions for"] = modal_vars


        # create the warning string
        for n,k in enumerate(sorted(warnings)):
            msg += '[{}] {}: {}.{}'.format(n+1,k,', '.join(warnings[k]), end_of_line)

        return msg

    def _detect_categorical_columns(self,data):
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
            likely_flag = 1.0 * data[var].nunique()/data[var].count() < 0.05
            if likely_flag:
                 likely_cat.append(var)
        return likely_cat

    def _q25(self,x):
        """
        Compute percentile (25th)
        """
        return np.nanpercentile(x.values,25)

    def _q75(self,x):
        """
        Compute percentile (75th)
        """
        return np.nanpercentile(x.values,75)

    def _std(self,x):
        """
        Compute standard deviation with ddof degrees of freedom
        """
        return np.nanstd(x.values,ddof=self._ddof)

    def _diptest(self,x):
        """
        Compute Hartigan Dip Test for modality.

        p < 0.05 suggests possible multimodality.
        """
        p = modality.hartigan_diptest(x.values)
        # dropna=False argument in pivot_table does not function as expected
        # return -1 instead of None
        if pd.isnull(p):
            return -1
        return p

    def _normaltest(self,x):
        """
        Compute test for normal distribution.

        Null hypothesis: x comes from a normal distribution
        p < alpha suggests the null hypothesis can be rejected.    
        """
        if len(x.values[~np.isnan(x.values)]) > 10:
            stat,p = stats.normaltest(x.values, nan_policy='omit')
        else:
            p = None
        # dropna=False argument in pivot_table does not function as expected
        # return -1 instead of None
        if pd.isnull(p):
            return -1
        return p

    def _tukey(self,x,threshold):
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
        except:
            outliers = []
        return outliers

    def _outliers(self,x):
        """
        Compute number of outliers
        """
        outliers = self._tukey(x, threshold = 1.5)
        return np.size(outliers)

    def _far_outliers(self,x):
        """
        Compute number of "far out" outliers
        """
        outliers = self._tukey(x, threshold = 3.0)
        return np.size(outliers)

    def _t1_summary(self,x):
        """
        Compute median [IQR] or mean (Std) for the input series.

        Parameters
        ----------
            x : pandas Series
                Series of values to be summarised.
        """
        if x.name in self._nonnormal:
            return '{:.2f} [{:.2f},{:.2f}]'.format(np.nanmedian(x.values),
                np.nanpercentile(x.values,25), np.nanpercentile(x.values,75))
        else:
            return '{:.2f} ({:.2f})'.format(np.nanmean(x.values),
                np.nanstd(x.values,ddof=self._ddof))

    def _create_cont_describe(self,data):
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
        aggfuncs = [pd.Series.count,np.mean,np.median,self._std,
            self._q25,self._q75,min,max,self._t1_summary,self._diptest,
            self._outliers,self._far_outliers,self._normaltest]

        # coerce continuous data to numeric
        cont_data = data[self._continuous].apply(pd.to_numeric, errors='coerce')
        # check all data in each continuous column is numeric
        bad_cols = cont_data.count() != data[self._continuous].count()
        bad_cols = cont_data.columns[bad_cols]
        if len(bad_cols)>0:
            raise InputError("""The following continuous column(s) have non-numeric values: {}.
            Either specify the column(s) as categorical or remove the non-numeric values.""".format(bad_cols.values))

        # check for coerced column containing all NaN to warn user
        for column in cont_data.columns[cont_data.count() == 0]:
            self._non_continuous_warning(column)

        if self._groupby:
            # add the groupby column back
            cont_data = cont_data.merge(data[[self._groupby]],
                left_index=True, right_index=True)

            # group and aggregate data
            df_cont = pd.pivot_table(cont_data,
                columns=[self._groupby],
                aggfunc=aggfuncs)
        else:
            # if no groupby, just add single group column
            df_cont = cont_data.apply(aggfuncs).T
            df_cont.columns.name = 'overall'
            df_cont.columns = pd.MultiIndex.from_product([df_cont.columns,
                ['overall']])

        df_cont.index.rename('variable',inplace=True)

        # remove prefix underscore from column names (e.g. _std -> std)
        agg_rename = df_cont.columns.levels[0]
        agg_rename = [x[1:] if x[0]=='_' else x for x in agg_rename]
        df_cont.columns.set_levels(agg_rename, level=0, inplace=True)

        return df_cont

    def _create_cat_describe(self,data):
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

        for g in self._groupbylvls:
            if self._groupby:
                d_slice = data.loc[data[self._groupby] == g, self._categorical]
            else:
                d_slice = data[self._categorical].copy()

            # create a dataframe with freq, proportion
            df = d_slice.copy()
            df = df.melt().groupby(['variable','value']).size().to_frame(name='freq')
            df.index.set_names('level', level=1, inplace=True)
            df['percent'] = df['freq'].div(df.freq.sum(level=0),level=0)* 100

            # add n column, listing total non-null values for each variable
            ct = d_slice.count().to_frame(name='n')
            ct.index.name = 'variable'
            df = df.join(ct)

            # add null count
            nulls = d_slice.isnull().sum().to_frame(name='isnull')
            nulls.index.name = 'variable'
            # only save null count to the first category for each variable
            # do this by extracting the first category from the df row index
            levels = df.reset_index()[['variable','level']].groupby('variable').first()
            # add this category to the nulls table
            nulls = nulls.join(levels)
            nulls.set_index('level', append=True, inplace=True)
            # join nulls to categorical
            df = df.join(nulls)

            # add summary column
            df['t1_summary'] = df.freq.map(str) + ' (' + df.percent.apply(round,
                ndigits=2).map(str) + ')'

            # add to dictionary
            group_dict[g] = df

        df_cat = pd.concat(group_dict,axis=1)
        # ensure the groups are the 2nd level of the column index
        if df_cat.columns.nlevels>1:
            df_cat = df_cat.swaplevel(0, 1, axis=1).sort_index(axis=1,level=0)

        return df_cat

    def _create_significance_table(self,data):
        """
        Create a table containing p-values for significance tests. Add features of
        the distributions and the p-values to the dataframe.

        Parameters
        ----------
            data : pandas DataFrame
                The input dataset.

        Returns
        ----------
            df : pandas DataFrame
                A table containing the p-values, test name, etc.
        """
        # list features of the variable e.g. matched, paired, n_expected
        df=pd.DataFrame(index=self._continuous+self._categorical,
            columns=['continuous','nonnormal','min_observed','pval','ptest'])

        df.index.rename('variable', inplace=True)
        df['continuous'] = np.where(df.index.isin(self._continuous),True,False)
        df['nonnormal'] = np.where(df.index.isin(self._nonnormal),True,False)

        # list values for each variable, grouped by groupby levels
        for v in df.index:
            is_continuous = df.loc[v]['continuous']
            is_categorical = ~df.loc[v]['continuous']
            is_normal = ~df.loc[v]['nonnormal']

            # if continuous, group data into list of lists
            if is_continuous:
                catlevels = None
                grouped_data = []
                for s in self._groupbylvls:
                    lvl_data = data.loc[data[self._groupby]==s, v]
                    # coerce to numeric and drop non-numeric data
                    lvl_data = lvl_data.apply(pd.to_numeric, errors='coerce').dropna()
                    # append to overall group data
                    grouped_data.append(lvl_data.values)
                min_observed = len(min(grouped_data,key=len))
            # if categorical, create contingency table
            elif is_categorical:
                catlevels = sorted(data[v].astype('category').cat.categories)
                grouped_data = pd.crosstab(data[self._groupby].rename('_groupby_var_'),data[v])
                min_observed = grouped_data.sum(axis=1).min()

            # minimum number of observations across all levels
            df.loc[v,'min_observed'] = min_observed

            # compute pvalues
            df.loc[v,'pval'],df.loc[v,'ptest'] = self._p_test(v,
                grouped_data,is_continuous,is_categorical,
                is_normal,min_observed,catlevels)

        return df

    def _p_test(self,v,grouped_data,is_continuous,is_categorical,
            is_normal,min_observed,catlevels):
        """
        Compute p-values.

        Parameters
        ----------
            v : str
                Name of the variable to be tested.
            grouped_data : list
                List of lists of values to be tested.
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
                The computed p-value.
            ptest : str
                The name of the test used to compute the p-value.
        """

        # no test by default
        pval=np.nan
        ptest='Not tested'

        # do not test if the variable has no observations in a level
        if min_observed == 0:
            warnings.warn('No p-value was computed for {} due to the low number of observations.'.format(v))
            return pval,ptest

        # continuous
        if is_continuous and is_normal and len(grouped_data)==2 :
            ptest = 'Two Sample T-test'
            test_stat, pval = stats.ttest_ind(*grouped_data,equal_var=False)
        elif is_continuous and is_normal:
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
                    ptest = "Fisher's exact"
                    oddsratio, pval = stats.fisher_exact(grouped_data)
                else:
                    ptest = 'Chi-squared (warning: expected count < 5)'
                    warnings.warn('No p-value was computed for {} due to the low number of observations.'.format(v))

        return pval,ptest

    def _create_cont_table(self,data):
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
        nulltable = data[self._continuous].isnull().sum().to_frame(name='isnull')
        try:
            table = table.join(nulltable)
        except TypeError: # if columns form a CategoricalIndex, need to convert to string first
            table.columns = table.columns.astype(str)
            table = table.join(nulltable)

        # add an empty level column, for joining with cat table
        table['level'] = ''
        table.set_index([table.index,'level'],inplace=True)

        # add pval column
        if self._pval and self._pval_adjust:
            table = table.join(self._significance_table[['pval (adjusted)','ptest']])
        elif self._pval:
            table = table.join(self._significance_table[['pval','ptest']])

        return table

    def _create_cat_table(self,data):
        """
        Create table one for categorical data.

        Returns
        ----------
        table : pandas DataFrame
            A table summarising the categorical variables.
        """
        table = self.cat_describe['t1_summary'].copy()
        # add the total count of null values across all levels
        isnull = data[self._categorical].isnull().sum().to_frame(name='isnull')
        isnull.index.rename('variable', inplace=True)
        try:
            table = table.join(isnull)
        except TypeError: # if columns form a CategoricalIndex, need to convert to string first
            table.columns = table.columns.astype(str)
            table = table.join(isnull)

        # add pval column
        if self._pval and self._pval_adjust:
            table = table.join(self._significance_table[['pval (adjusted)','ptest']])
        elif self._pval:
            table = table.join(self._significance_table[['pval','ptest']])

        return table

    def _create_tableone(self,data):
        """
        Create table 1 by combining the continuous and categorical tables.

        Returns
        ----------
        table : pandas DataFrame
            The complete table one.
        """
        if self._continuous and self._categorical:
            table = pd.concat([self.cont_table,self.cat_table],sort=False)
        elif self._continuous:
            table = self.cont_table
        elif self._categorical:
            table = self.cat_table

        # round pval column and convert to string
        if self._pval and self._pval_adjust:
            table['pval (adjusted)'] = table['pval (adjusted)'].apply('{:.3f}'.format).astype(str)
            table.loc[table['pval (adjusted)'] == '0.000', 'pval (adjusted)'] = '<0.001'
        elif self._pval:
            table['pval'] = table['pval'].apply('{:.3f}'.format).astype(str)
            table.loc[table['pval'] == '0.000', 'pval'] = '<0.001'

        # sort the table rows
        table.reset_index().set_index(['variable','level'], inplace=True)
        if self._sort:
            # alphabetical
            new_index = sorted(table.index.values)
        else:
            # sort by the columns argument
            new_index = sorted(table.index.values,key=lambda x: self._columns.index(x[0]))
        table = table.reindex(new_index)

        # if a limit has been set on the number of categorical variables
        # then re-order the variables by frequency
        if self._limit:
            levelcounts = data[self._categorical].nunique()
            levelcounts = levelcounts[levelcounts >= self._limit]
            for v,_ in levelcounts.iteritems():
                count = data[v].value_counts().sort_values(ascending=False)
                new_index = [(v, i) for i in count.index]
                # restructure to match orig_index
                new_index_array=np.empty((len(new_index),), dtype=object)
                new_index_array[:]=[tuple(i) for i in new_index]
                orig_index = table.index.values.copy()
                orig_index[table.index.get_loc(v)] = new_index_array
                table = table.reindex(orig_index)

        # inserts n row
        n_row = pd.DataFrame(columns = ['variable','level','isnull'])
        n_row.set_index(['variable','level'], inplace=True)
        n_row.loc['n', ''] = None
        table = pd.concat([n_row,table],sort=False)

        if self._groupbylvls == ['overall']:
            table.loc['n','overall'] = len(data.index)
        else:
            for g in self._groupbylvls:
                ct = data[self._groupby][data[self._groupby]==g].count()
                table.loc['n',g] = ct

        # only display data in first level row
        dupe_mask = table.groupby(level=[0]).cumcount().ne(0)
        dupe_columns = ['isnull']
        optional_columns = ['pval','pval (adjusted)','ptest']
        for col in optional_columns:
            if col in table.columns.values:
                dupe_columns.append(col)

        table[dupe_columns] = table[dupe_columns].mask(dupe_mask).fillna('')

        # remove empty column added above
        table.drop([''], axis=1, inplace=True)

        # remove isnull column if not needed
        if not self._isnull:
            table.drop('isnull',axis=1,inplace=True)

        # replace nans with empty strings
        table.fillna('',inplace=True)

        # add column index
        if not self._groupbylvls == ['overall']:
            # rename groupby variable if requested
            c = self._groupby
            if self._labels:
                if self._groupby in self._labels:
                    c = self._labels[self._groupby]

            c = 'Grouped by {}'.format(c)
            table.columns = pd.MultiIndex.from_product([[c], table.columns])

        # display alternative labels if assigned
        if self._labels:
            table.rename(index=self._labels, inplace=True, level=0)

        # if a limit has been set on the number of categorical variables
        # limit the number of categorical variables that are displayed
        if self._limit:
            table = table.groupby('variable').head(self._limit)

        # re-order the columns in a consistent fashion
        if self._groupby:
            cols = table.columns.levels[1].values
        else:
            cols = table.columns.values

        if 'isnull' in cols:
            cols = ['isnull'] + [x for x in cols if x != 'isnull']

        # iterate through each optional column
        # if they exist, put them at the end of the dataframe
        # ensures the last 3 columns will be in the same order as optional_columns
        for col in optional_columns:
            if col in cols:
                cols = [x for x in cols if x != col] + [col]

        if self._groupby:
            table = table.reindex(cols, axis=1, level=1)
        else:
            table = table.reindex(cols, axis=1)

        return table

    # warnings
    def _non_continuous_warning(self, c):
        warnings.warn('"{}" has all non-numeric values. Consider including it in the list of categorical variables.'.format(c), RuntimeWarning, stacklevel=2)
