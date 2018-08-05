import pandas as pd
import tableone
from tableone import TableOne
from tableone import InputError
from nose.tools import with_setup, assert_raises, assert_equal
import numpy as np
import modality
import warnings

class TestTableOne(object):
    """
    Tests for TableOne
    """

    def setup(self):
        """
        set up test fixtures
        """
        seed = 12345
        np.random.seed(seed)
        self.data_pn = self.create_pn_dataset()
        self.data_sample = self.create_sample_dataset(n = 10000)
        self.data_small = self.create_small_dataset()
        self.data_groups = self.create_another_dataset(n = 20)
        self.data_categorical = self.create_categorical_dataset()
        self.data_mixed = self.create_mixed_datatypes_dataset()

    def create_pn_dataset(self):
        """
        create pn dataset
        """
        url="https://raw.githubusercontent.com/tompollard/tableone/master/data/pn2012_demo.csv"
        return pd.read_csv(url)

    def create_sample_dataset(self, n):
        """
        create sample dataset
        """
        data_sample = pd.DataFrame(index=range(n))

        mu, sigma = 10, 1
        data_sample['normal'] = np.random.normal(mu, sigma, n)
        data_sample['nonnormal'] = np.random.noncentral_chisquare(20,nonc=2,size=n)

        bears = ['Winnie','Paddington','Baloo','Blossom']
        data_sample['bear'] = np.random.choice(bears, n, p=[0.5, 0.1, 0.1, 0.3])

        data_sample['likeshoney'] = np.nan
        data_sample.loc[data_sample['bear'] == 'Winnie', 'likeshoney'] = 1
        data_sample.loc[data_sample['bear'] == 'Baloo', 'likeshoney'] = 1

        data_sample['likesmarmalade'] = 0
        data_sample.loc[data_sample['bear'] == 'Paddington', 'likesmarmalade'] = 1

        data_sample['height'] = 0
        data_sample.loc[data_sample['bear'] == 'Winnie', 'height'] = 6
        data_sample.loc[data_sample['bear'] == 'Paddington', 'height'] = 4
        data_sample.loc[data_sample['bear'] == 'Baloo', 'height'] = 20
        data_sample.loc[data_sample['bear'] == 'Blossom', 'height'] = 7

        data_sample['fictional'] = 0
        data_sample.loc[data_sample['bear'] == 'Winnie', 'fictional'] = 1
        data_sample.loc[data_sample['bear'] == 'Paddington', 'fictional'] = 1
        data_sample.loc[data_sample['bear'] == 'Baloo', 'fictional'] = 1
        data_sample.loc[data_sample['bear'] == 'Blossom', 'fictional'] = 1

        return data_sample

    def create_small_dataset(self):
        """
        create small dataset
        """
        data_small = pd.DataFrame(index=range(10))
        data_small['group1'] = 0
        data_small.loc[0:4, 'group1'] = 1
        data_small['group2'] = 0
        data_small.loc[2:7, 'group2'] = 1
        data_small['group3'] = 0
        data_small.loc[1:2, 'group3'] = 1
        data_small.loc[3:7, 'group3'] = 2

        return data_small

    def create_another_dataset(self, n):
        """
        create another dataset
        """
        data_groups = pd.DataFrame(index=range(n))
        data_groups['group'] = 'group1'
        data_groups.loc[ 2:6, 'group'] = 'group2'
        data_groups.loc[ 6:12, 'group'] = 'group3'
        data_groups.loc[12: n, 'group'] = 'group4'
        data_groups['age'] = range(n)
        data_groups['weight'] = [x+100 for x in range(n)]

        return data_groups

    def create_categorical_dataset(self, n_cat=100, n_obs_per_cat=1000, n_col=10):
        """
        create a dataframe with many categories of many levels
        """
        # dataframe with many categories of many levels
        # generate integers to represent data
        data = np.arange(n_cat*n_obs_per_cat*n_col)
        # use modulus to create categories - unique for each column
        data = np.mod(data,n_cat*n_col)
        # reshape intro a matrix
        data = data.reshape(n_cat*n_obs_per_cat, n_col)
        return pd.DataFrame(data)

    def create_mixed_datatypes_dataset(self, n=20):
        """
        create a dataframe with mixed datatypes in the same column
        """
        data_mixed = pd.DataFrame(index=range(n))

        data_mixed['string data'] = 'a'

        mu, sigma = 50, 5
        data_mixed['mixed numeric data'] = np.random.normal(mu, sigma, n)
        data_mixed.loc[1, 'mixed numeric data'] = 'could not measure'
        return data_mixed

    def teardown(self):
        """
        tear down test fixtures
        """
        pass

    @with_setup(setup, teardown)
    def test_hello_travis(self):
        x = 'hello'
        y = 'travis'
        assert x != y

    @with_setup(setup, teardown)
    def test_examples_used_in_the_readme_run_without_raising_error_pn(self):

        columns = ['Age', 'SysABP', 'Height', 'Weight', 'ICU', 'death']
        categorical = ['ICU', 'death']
        groupby = ['death']
        nonnormal = ['Age']
        mytable = TableOne(self.data_pn, columns=columns, categorical=categorical,
            groupby=groupby, nonnormal=nonnormal, pval=False)

    @with_setup(setup, teardown)
    def test_overall_mean_and_std_as_expected_for_cont_variable(self):

        columns=['normal','nonnormal','height']
        table = TableOne(self.data_sample, columns=columns)

        mean =  table.cont_describe.loc['normal']['mean']['overall']
        std = table.cont_describe.loc['normal']['std']['overall']

        print(self.data_sample.mean())
        print(self.data_sample.std())

        assert abs(mean-self.data_sample.normal.mean()) <= 0.02
        assert abs(std-self.data_sample.normal.std()) <= 0.02

    @with_setup(setup, teardown)
    def test_overall_n_and_percent_as_expected_for_binary_cat_variable(self):

        categorical=['likesmarmalade']
        table = TableOne(self.data_sample, columns=categorical, categorical=categorical)

        lm = table.cat_describe.loc['likesmarmalade']
        # drop 2nd level for convenience
        lm.columns = lm.columns.droplevel(level=1)
        notlikefreq = lm.loc[0,'freq']
        notlikepercent = lm.loc[0,'percent']
        likefreq = lm.loc[1,'freq']
        likepercent = lm.loc[1,'percent']

        assert notlikefreq + likefreq == 10000
        assert abs(100 - notlikepercent - likepercent) <= 0.02
        assert notlikefreq == 8977
        assert likefreq == 1023

    @with_setup(setup, teardown)
    def test_overall_n_and_percent_as_expected_for_binary_cat_variable_with_nan(self):
        """
        Ignore NaNs when counting the number of values and the overall percentage
        """
        categorical=['likeshoney']
        table = TableOne(self.data_sample, columns=categorical, categorical=categorical)

        lh = table.cat_describe.loc['likeshoney']
        # drop 2nd level for convenience
        lh.columns = lh.columns.droplevel(level=1)
        likefreq = lh.loc[1.0,'freq']
        likepercent = lh.loc[1.0,'percent']

        assert likefreq == 5993
        assert abs(100-likepercent) <= 0.01

    @with_setup(setup, teardown)
    def test_with_data_as_only_input_argument(self):
        """
        Test with a simple dataset that a table generated with no pre-specified columns
        returns the same results as a table generated with specified columns
        """
        table_no_args = TableOne(self.data_groups)

        columns = ['group','age','weight']
        categorical=['group']
        table_with_args = TableOne(self.data_groups, columns=columns, categorical=categorical)

        assert table_no_args._columns == table_with_args._columns
        assert table_no_args._categorical == table_with_args._categorical
        assert table_no_args._remarks == table_with_args._remarks
        assert (table_no_args.tableone.columns == table_with_args.tableone.columns).all()
        assert (table_no_args.tableone['overall'].values == \
            table_with_args.tableone['overall'].values).all()
        assert (table_no_args.tableone == table_with_args.tableone).all().all()

    @with_setup(setup, teardown)
    def test_fisher_exact_for_small_cell_count(self):
        """
        Ensure that the package runs Fisher exact if cell counts are <=5 and it's a 2x2
        """
        categorical=['group1','group3']
        table = TableOne(self.data_small, categorical=categorical, groupby='group2', 
            pval=True)

        # group2 should be tested because it's a 2x2
        # group3 is a 2x3 so should not be tested
        assert table._significance_table.loc['group1','ptest'] == "Fisher's exact"
        assert table._significance_table.loc['group3','ptest'] == \
            'Chi-squared (warning: expected count < 5)'

    @with_setup(setup, teardown)
    def test_sequence_of_cont_table(self):
        """
        Ensure that the columns align with the values
        """
        columns = ['age','weight']
        categorical = []
        groupby = 'group'
        t = TableOne(self.data_groups, columns = columns,
            categorical = categorical, groupby = groupby, isnull = False)

        # n and weight rows are already ordered, so sorting should not alter the order
        assert (t.tableone.loc['n'].values[0].astype(float) == \
            sorted(t.tableone.loc['n'].values[0].astype(float))).any()
        assert (t.tableone.loc['age'].values[0] == \
            ['0.50 (0.71)', '3.50 (1.29)', '8.50 (1.87)', '15.50 (2.45)']).any()

    @with_setup(setup, teardown)
    def test_categorical_cell_count(self):
        """
        Check the categorical cell counts are correct
        """
        categorical=list(np.arange(10))
        table = TableOne(self.data_categorical, columns=categorical,categorical=categorical)
        df = table.cat_describe
        # drop 'overall' level of column index
        df.columns = df.columns.droplevel(level=1)
        # each column
        for i in np.arange(10):
            # each category should have 100 levels
            assert df.loc[i].shape[0] == 100

    @with_setup(setup, teardown)
    def test_hartigan_diptest_for_modality(self):
        """
        Ensure that the package runs Fisher exact if cell counts are <=5 and it's a 2x2
        """
        dist_1_peak = modality.generate_data(peaks=1, n=[10000])
        t1=modality.hartigan_diptest(dist_1_peak)
        assert t1 > 0.95

        dist_2_peak = modality.generate_data(peaks=2, n=[10000, 10000])
        t2=modality.hartigan_diptest(dist_2_peak)
        assert t2 < 0.05

        dist_3_peak = modality.generate_data(peaks=3, n=[10000, 10000, 10000])
        t3=modality.hartigan_diptest(dist_3_peak)
        assert t3 < 0.05

    @with_setup(setup, teardown)
    def test_limit_of_categorical_data_pn(self):
        """
        Tests the `limit` keyword arg, which limits the number of categories presented
        """
        data_pn = self.data_pn.copy()
        # 6 categories of age based on decade
        data_pn['age_group'] = data_pn['Age'].map(lambda x: int(x/10))

        # limit 
        columns = ['age_group', 'Age', 'SysABP', 'Height', 'Weight', 'ICU', 'death']
        categorical = ['age_group','ICU', 'death']

        # test it limits to 3
        table = TableOne(data_pn, columns=columns, categorical=categorical, limit=3)
        assert table.tableone.loc['age_group',:].shape[0] == 3

        # test other categories are not affected if limit > num categories
        assert table.tableone.loc['death',:].shape[0] == 2

    def test_input_data_not_modified(self):
        """
        Check the input dataframe is not modified by the package
        """
        df_orig = self.data_groups.copy()

        # turn off warnings for this test
        # warnings.simplefilter("ignore")

        # no input arguments
        df_no_args = self.data_groups.copy()
        table_no_args = TableOne(df_no_args)
        assert (df_no_args['group'] == df_orig['group']).all()

        # groupby
        df_groupby = self.data_groups.copy()
        table_groupby = TableOne(df_groupby, columns = ['group','age','weight'],
            categorical = ['group'], groupby=['group'])
        assert (df_groupby['group'] == df_orig['group']).all()
        assert (df_groupby['age'] == df_orig['age']).all()
        assert (df_groupby['weight'] == df_orig['weight']).all()

        # sorted
        df_sorted = self.data_groups.copy()
        table_sorted = TableOne(df_sorted, columns = ['group','age','weight'],
            categorical = ['group'], groupby=['group'], sort=True)
        assert (df_sorted['group'] == df_orig['group']).all()
        assert (df_groupby['age'] == df_orig['age']).all()
        assert (df_groupby['weight'] == df_orig['weight']).all()

        # pval
        df_pval = self.data_groups.copy()
        table_pval = TableOne(df_pval, columns = ['group','age','weight'],
            categorical = ['group'], groupby=['group'], sort=True, pval=True)
        assert (df_pval['group'] == df_orig['group']).all()
        assert (df_groupby['age'] == df_orig['age']).all()
        assert (df_groupby['weight'] == df_orig['weight']).all()

        # pval_adjust
        df_pval_adjust = self.data_groups.copy()
        table_pval_adjust = TableOne(df_pval_adjust, columns = ['group','age','weight'],
            categorical = ['group'], groupby=['group'], sort=True, pval=True,
            pval_adjust='bonferroni')
        assert (df_pval_adjust['group'] == df_orig['group']).all()
        assert (df_groupby['age'] == df_orig['age']).all()
        assert (df_groupby['weight'] == df_orig['weight']).all()

        # labels
        df_labels = self.data_groups.copy()
        table_labels = TableOne(df_labels, columns = ['group','age','weight'],
            categorical = ['group'], groupby=['group'], labels={'age':'age, years'})
        assert (df_labels['group'] == df_orig['group']).all()
        assert (df_groupby['age'] == df_orig['age']).all()
        assert (df_groupby['weight'] == df_orig['weight']).all()

        # limit
        df_limit = self.data_groups.copy()
        table_limit = TableOne(df_limit, columns = ['group','age','weight'],
            categorical = ['group'], groupby=['group'], limit=2)
        assert (df_limit['group'] == df_orig['group']).all()
        assert (df_groupby['age'] == df_orig['age']).all()
        assert (df_groupby['weight'] == df_orig['weight']).all()

        # nonnormal
        df_nonnormal = self.data_groups.copy()
        table_nonnormal = TableOne(df_nonnormal, columns = ['group','age','weight'],
            categorical = ['group'], groupby=['group'], nonnormal=['age'])
        assert (df_nonnormal['group'] == df_orig['group']).all()
        assert (df_groupby['age'] == df_orig['age']).all()
        assert (df_groupby['weight'] == df_orig['weight']).all()

        # warnings.simplefilter("default")

    @with_setup(setup, teardown)
    def test_groupby_with_group_named_isnull_pn(self):
        """
        Test case with a group having the same name as a column in TableOne
        """
        df = self.data_pn.copy()

        columns = ['Age', 'SysABP', 'Height', 'Weight', 'ICU']
        groupby = 'ICU'
        group_levels = df[groupby].unique()

        # collect the possible column names
        table = TableOne(df, columns=columns, groupby=groupby, pval=True)
        tableone_columns = list(table.tableone.columns.levels[1])

        table = TableOne(df, columns=columns, groupby=groupby, pval=True, pval_adjust='b')
        tableone_columns = tableone_columns + list(table.tableone.columns.levels[1])
        tableone_columns = np.unique(tableone_columns)
        tableone_columns = [c for c in tableone_columns if c not in group_levels]

        for c in tableone_columns:
            # for each output column name in tableone, try them as a group
            df.loc[0:20,'ICU'] = c
            if 'adjust' in c:
                pval_adjust='b'
            else:
                pval_adjust=None

            with assert_raises(InputError):
                table = TableOne(df, columns=columns, groupby=groupby, pval=True, 
                    pval_adjust=pval_adjust)

    @with_setup(setup, teardown)
    def test_label_dictionary_input_pn(self):
        """
        Test columns and rows are relabelled with the label argument
        """
        df = self.data_pn.copy()
        columns = ['Age', 'ICU','death']
        categorical = ['death','ICU']
        groupby = 'death'

        labels = {'death': 'mortality', 'Age': 'Age, years', 
        'ICU': 'Intensive Care Unit'}

        table = TableOne(df, columns=columns, categorical=categorical, groupby=groupby, 
            labels=labels)

        # check the header column is updated (groupby variable)
        assert table.tableone.columns.levels[0][0] == 'Grouped by mortality'

        # check the categorical rows are updated
        assert 'Intensive Care Unit' in table.tableone.index.levels[0]

        # check the continuous rows are updated
        assert 'Age, years' in table.tableone.index.levels[0]

    @with_setup(setup, teardown)
    def test_tableone_row_sort_pn(self):
        """
        Test sort functionality of TableOne
        """
        df = self.data_pn.copy()
        columns = ['Age', 'SysABP', 'Height', 'Weight', 'ICU', 'death']
        table = TableOne(df, columns=columns)

        # a call to .index.levels[0] automatically sorts the levels
        # instead, call values and use pd.unique as it preserves order
        tableone_rows = pd.unique([x[0] for x in table.tableone.index.values])

        # default should not sort
        for i, c in enumerate(columns):
            # i+1 because we skip the first row, 'n'
            assert tableone_rows[i+1] == c

        table = TableOne(df, columns=columns, sort=True)
        tableone_rows = pd.unique([x[0] for x in table.tableone.index.values])
        for i, c in enumerate(np.sort(columns)):
            # i+1 because we skip the first row, 'n'
            assert tableone_rows[i+1] == c

    @with_setup(setup, teardown)
    def test_string_data_as_continuous_error(self):
        """
        Test raising an error when continuous columns contain non-numeric data
        """
        try:
            # Trigger the categorical warning
            table = TableOne(self.data_mixed, categorical=[])
        except InputError as e:
            starts_str = "The following continuous column(s) have non-numeric values"
            assert e.args[0].startswith(starts_str)
        except:
            # unexpected error - raise it
            raise

    @with_setup(setup, teardown)
    def test_tableone_columns_in_consistent_order_pn(self):
        """
        Test output columns in TableOne are always in the same order
        """
        df = self.data_pn.copy()
        columns = ['Age', 'SysABP', 'Height', 'Weight', 'ICU', 'death']
        categorical = ['ICU', 'death']
        groupby = ['death']

        table = TableOne(df, columns=columns, groupby=groupby, pval=True)

        assert table.tableone.columns.levels[1][0] == 'isnull'
        assert table.tableone.columns.levels[1][-1] == 'ptest'
        assert table.tableone.columns.levels[1][-2] == 'pval'

        df.loc[df['death']==0, 'death'] = 2
        table = TableOne(df, columns=columns, groupby=groupby, pval=True, 
            pval_adjust='bonferroni')

        assert table.tableone.columns.levels[1][0] == 'isnull'
        assert table.tableone.columns.levels[1][-1] == 'ptest'
        assert table.tableone.columns.levels[1][-2] == 'pval (adjusted)'
        table

    @with_setup(setup, teardown)
    def test_check_null_counts_are_correct_pn(self):
        """
        Test that the isnull column is correctly reporting number of nulls
        """
        columns = ['Age', 'SysABP', 'Height', 'Weight', 'ICU', 'death']
        categorical = ['ICU', 'death']
        groupby = ['death']

        # test when not grouping
        table = TableOne(self.data_pn, columns=columns, categorical=categorical)

        # get isnull column only
        isnull = table.tableone.iloc[:,0]
        for i, v in enumerate(isnull):
            # skip empty rows by checking value is not a string
            if 'float' in str(type(v)):
                # check each null count is correct
                col = isnull.index[i][0]
                assert self.data_pn[col].isnull().sum() == v

        # test when grouping by a variable
        grouped_table = TableOne(self.data_pn, columns=columns, 
            categorical=categorical, groupby=groupby)

        # get isnull column only
        isnull = grouped_table.tableone.iloc[:,0]
        for i, v in enumerate(isnull):
            # skip empty rows by checking value is not a string
            if 'float' in str(type(v)):
                # check each null count is correct
                col = isnull.index[i][0]
                assert self.data_pn[col].isnull().sum() == v

    @with_setup(setup, teardown)
    def test_multilevel_groupby(self):
        """
        Test multilevel groupby produces expected results
        """
        columns = ['Age', 'Height', 'Weight', 'ICU']
        categorical = ['ICU']

        table = TableOne(self.data_pn, columns=columns, categorical=categorical, groupby=['death', 'MechVent'])
        assert table.tableone.columns[0][0] == 'Grouped by death, MechVent'
        table.tableone.columns = table.tableone.columns.droplevel(0)
        assert len(table.tableone.columns) == 5
        for i, correct_col in enumerate([('isnull', ''), ('0', '0'), ('0', '1'), ('1', '0'), ('1', '1')]):
                assert table.tableone.columns[i] == correct_col
        assert len(table.tableone.index) == 8
        rows = [('n', ''), ('Age', ''), ('Height', ''), ('Weight', ''), ('ICU', 'CCU'), ('ICU', 'CSRU'), ('ICU', 'MICU'), ('ICU', 'SICU')]
        for i, correct_row in enumerate(rows):
                assert table.tableone.index[i] == correct_row
        correct_value = {
                ('n', ''): ['', 468, 396, 72, 64],
                ('Age', ''): [0, '65.29 (17.94)', '62.47 (16.65)', '71.06 (13.90)', '72.42 (14.21)'],
                ('Height', ''): [475, '171.55 (31.78)', '169.24 (11.06)', '167.36 (11.32)', '169.86 (11.34)'],
                ('Weight', ''): [302, '81.03 (22.28)', '85.02 (24.67)', '83.89 (28.35)', '80.44 (21.66)'],
                ('ICU', 'CCU'): [0, '110 (23.5)', '27 (6.82)', '11 (15.28)', '14 (21.88)'],
                ('ICU', 'CSRU'): ['', '50 (10.68)', '144 (36.36)', '3 (4.17)', '5 (7.81)'],
                ('ICU', 'MICU'): ['', '205 (43.8)', '113 (28.54)', '47 (65.28)', '15 (23.44)'],
                ('ICU', 'SICU'): ['', '103 (22.01)', '112 (28.28)', '11 (15.28)', '30 (46.88)']
        }
        for row in rows:
                assert list(table.tableone.loc[row]) == correct_value[row]

    @with_setup(setup, teardown)
    def test_multilevel_groupby_pval(self):
        """
        Test multilevel groupby works when p-values are requested
        """
        columns = ['Age', 'Height', 'Weight', 'ICU']
        categorical = ['ICU']

        table = TableOne(self.data_pn, columns=columns, categorical=categorical, groupby=['death', 'MechVent'], pval=True)
        table = TableOne(self.data_pn, columns=columns, categorical=categorical, groupby=['death', 'MechVent'], pval=True, pval_adjust='bonferroni')
        table = TableOne(self.data_pn, columns=columns, categorical=categorical, groupby=['death', 'MechVent'], pval=True, nonnormal=['Age'])
        assert table.tableone.loc['Weight', ('Grouped by death, MechVent', 'pval', '')][0] == '0.187'

    @with_setup(setup, teardown)
    def test_multilevel_groupby_noisnull(self):
        """
        Test multilevel groupby runs without error when isnull option is False
        """
        columns = ['Age', 'Height', 'Weight', 'ICU']
        categorical = ['ICU']

        table = TableOne(self.data_pn, columns=columns, categorical=categorical, groupby=['death', 'MechVent'], isnull=False)

    @with_setup(setup, teardown)
    def test_multilevel_groupby_sort(self):
        """
        Test multilevel groupby runs without error when sort option is True
        """
        columns = ['Age', 'Height', 'Weight', 'ICU']
        categorical = ['ICU']

        table = TableOne(self.data_pn, columns=columns, categorical=categorical, groupby=['death', 'MechVent'], sort=True)

    @with_setup(setup, teardown)
    def test_multilevel_groupby_limit(self):
        """
        Test multilevel groupby runs correctly when limit option is set
        """
        columns = ['Age', 'Height', 'Weight', 'ICU']
        categorical = ['ICU']

        table = TableOne(self.data_pn, columns=columns, categorical=categorical, groupby=['death', 'MechVent'], limit=2)
        assert list(table.tableone.loc['ICU'].index) == ['MICU', 'SICU']

    @with_setup(setup, teardown)
    def test_groupby_categorical(self):
        """
        Test groupby runs without error with categorical groupby variable
        """
        columns = ['Age', 'Height', 'Weight', 'ICU']
        categorical = ['ICU']

        pn = self.data_pn.copy()
        pn['death'] = pn['death'].astype('category')
        table = TableOne(pn, columns=columns, categorical=categorical, groupby=['death'])
        assert len(table.tableone.columns == 3)
        table = TableOne(pn, columns=columns, categorical=categorical, groupby=['death', 'MechVent'])
        assert len(table.tableone.columns) == 5

