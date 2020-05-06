import random
import warnings

from nose.tools import (with_setup, assert_raises, assert_equal,
                        assert_almost_equal)
import numpy as np
import pandas as pd
from scipy import stats

import tableone
from tableone import TableOne, load_dataset
from tableone.tableone import InputError
from tableone.modality import hartigan_diptest, generate_data


def mytest(*args):
    """
    Hypothesis test for test_self_defined_statistical_tests
    """
    mytest.__name__ = "Test name"
    _, pval = stats.ks_2samp(*args)
    return pval


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
        self.data_pn = load_dataset('pn2012')
        self.data_sample = self.create_sample_dataset(n=10000)
        self.data_small = self.create_small_dataset()
        self.data_groups = self.create_another_dataset(n=20)
        self.data_categorical = self.create_categorical_dataset()
        self.data_mixed = self.create_mixed_datatypes_dataset()

    def create_sample_dataset(self, n):
        """
        create sample dataset
        """
        data_sample = pd.DataFrame(index=range(n))

        mu, sigma = 10, 1
        data_sample['normal'] = np.random.normal(mu, sigma, n)
        data_sample['nonnormal'] = np.random.noncentral_chisquare(20, nonc=2,
                                                                  size=n)

        bears = ['Winnie', 'Paddington', 'Baloo', 'Blossom']
        data_sample['bear'] = np.random.choice(bears, n,
                                               p=[0.5, 0.1, 0.1, 0.3])

        data_sample['likeshoney'] = np.nan
        data_sample.loc[data_sample['bear'] == 'Winnie', 'likeshoney'] = 1
        data_sample.loc[data_sample['bear'] == 'Baloo', 'likeshoney'] = 1

        data_sample['likesmarmalade'] = 0
        data_sample.loc[data_sample['bear'] == 'Paddington',
                                               'likesmarmalade'] = 1

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
        data_groups.loc[2:6, 'group'] = 'group2'
        data_groups.loc[6:12, 'group'] = 'group3'
        data_groups.loc[12: n, 'group'] = 'group4'
        data_groups['age'] = range(n)
        data_groups['weight'] = [x+100 for x in range(n)]

        return data_groups

    def create_categorical_dataset(self, n_cat=100, n_obs_per_cat=1000,
                                   n_col=10):
        """
        create a dataframe with many categories of many levels
        """
        # dataframe with many categories of many levels
        # generate integers to represent data
        data = np.arange(n_cat*n_obs_per_cat*n_col)
        # use modulus to create categories - unique for each column
        data = np.mod(data, n_cat*n_col)
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
        mytable = TableOne(self.data_pn, columns=columns,
                           categorical=categorical, groupby=groupby,
                           nonnormal=nonnormal, pval=False)

    @with_setup(setup, teardown)
    def test_overall_mean_and_std_as_expected_for_cont_variable(self):

        columns = ['normal', 'nonnormal', 'height']
        table = TableOne(self.data_sample, columns=columns)

        mean = table.cont_describe.loc['normal']['mean']['Overall']
        std = table.cont_describe.loc['normal']['std']['Overall']

        print(self.data_sample.mean())
        print(self.data_sample.std())

        assert abs(mean-self.data_sample.normal.mean()) <= 0.02
        assert abs(std-self.data_sample.normal.std()) <= 0.02

    @with_setup(setup, teardown)
    def test_overall_n_and_percent_as_expected_for_binary_cat_variable(self):

        categorical = ['likesmarmalade']
        table = TableOne(self.data_sample, columns=categorical,
                         categorical=categorical)

        lm = table.cat_describe.loc['likesmarmalade']

        notlikefreq = float(lm.loc['0', 'freq'].values[0])
        notlikepercent = float(lm.loc['0', 'percent'].values[0])
        likefreq = float(lm.loc['1', 'freq'].values[0])
        likepercent = float(lm.loc['1', 'percent'].values[0])

        assert notlikefreq + likefreq == 10000
        assert abs(100 - notlikepercent - likepercent) <= 0.02
        assert notlikefreq == 8977
        assert likefreq == 1023

    @with_setup(setup, teardown)
    def test_overall_n_and_percent_for_binary_cat_var_with_nan(self):
        """
        Ignore NaNs when counting the number of values and the overall
        percentage
        """
        categorical = ['likeshoney']
        table = TableOne(self.data_sample, columns=categorical,
                         categorical=categorical)

        lh = table.cat_describe.loc['likeshoney']

        likefreq = float(lh.loc['1.0', 'freq'].values[0])
        likepercent = float(lh.loc['1.0', 'percent'].values[0])

        assert likefreq == 5993
        assert abs(100-likepercent) <= 0.01

    @with_setup(setup, teardown)
    def test_with_data_as_only_input_argument(self):
        """
        Test with a simple dataset that a table generated with no pre-specified
        columns returns the same results as a table generated with specified
        columns
        """
        table_no_args = TableOne(self.data_groups)

        columns = ['group', 'age', 'weight']
        categorical = ['group']
        table_with_args = TableOne(self.data_groups, columns=columns,
                                   categorical=categorical)

        assert table_no_args._columns == table_with_args._columns
        assert table_no_args._categorical == table_with_args._categorical
        assert table_no_args._remarks == table_with_args._remarks
        assert (table_no_args.tableone.columns ==
                table_with_args.tableone.columns).all()
        assert (table_no_args.tableone['Overall'].values ==
                table_with_args.tableone['Overall'].values).all()
        assert (table_no_args.tableone == table_with_args.tableone).all().all()

    @with_setup(setup, teardown)
    def test_fisher_exact_for_small_cell_count(self):
        """
        Ensure that the package runs Fisher exact if cell counts are <=5
        and it is a 2x2
        """
        categorical = ['group1', 'group3']
        table = TableOne(self.data_small, categorical=categorical,
                         groupby='group2', pval=True)

        # group2 should be tested because it's a 2x2
        # group3 is a 2x3 so should not be tested
        assert (table._htest_table.loc['group1', 'Test'] == "Fisher's exact")
        assert (table._htest_table.loc['group3', 'Test'] ==
                'Chi-squared (warning: expected count < 5)')

    @with_setup(setup, teardown)
    def test_sequence_of_cont_table(self):
        """
        Ensure that the columns align with the values
        """
        columns = ['age', 'weight']
        categorical = []
        groupby = 'group'
        t = TableOne(self.data_groups, columns=columns,
                     categorical=categorical, groupby=groupby,
                     missing=False, decimals=2, label_suffix=False,
                     overall=False)

        # n and weight rows are already ordered, so sorting should
        # not change the order
        assert (t.tableone.loc['n'].values[0].astype(float) ==
                sorted(t.tableone.loc['n'].values[0].astype(float))).any()
        assert (t.tableone.loc['age'].values[0] ==
                ['0.50 (0.71)', '3.50 (1.29)', '8.50 (1.87)',
                 '15.50 (2.45)']).any()

    @with_setup(setup, teardown)
    def test_categorical_cell_count(self):
        """
        Check the categorical cell counts are correct
        """
        categorical = list(np.arange(10))
        table = TableOne(self.data_categorical, columns=categorical,
                         categorical=categorical)
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
        Ensure that the package runs Fisher exact if cell counts are <=5
        and it is a 2x2
        """
        dist_1_peak = generate_data(peaks=1, n=[10000])
        t1 = hartigan_diptest(dist_1_peak)
        assert t1 > 0.95

        dist_2_peak = generate_data(peaks=2, n=[10000, 10000])
        t2 = hartigan_diptest(dist_2_peak)
        assert t2 < 0.05

        dist_3_peak = generate_data(peaks=3, n=[10000, 10000, 10000])
        t3 = hartigan_diptest(dist_3_peak)
        assert t3 < 0.05

    @with_setup(setup, teardown)
    def test_limit_of_categorical_data_pn(self):
        """
        Tests the `limit` keyword arg, which limits the number of categories
        presented
        """
        data_pn = self.data_pn.copy()
        # 6 categories of age based on decade
        data_pn['age_group'] = data_pn['Age'].map(lambda x: int(x/10))

        # limit
        columns = ['age_group', 'Age', 'SysABP', 'Height', 'Weight', 'ICU',
                   'death']
        categorical = ['age_group', 'ICU', 'death']

        # test it limits to 3
        table = TableOne(data_pn, columns=columns, categorical=categorical,
                         limit=3, label_suffix=False)
        assert table.tableone.loc['age_group', :].shape[0] == 3

        # test other categories are not affected if limit > num categories
        assert table.tableone.loc['death', :].shape[0] == 2

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
        table_groupby = TableOne(df_groupby,
                                 columns=['group', 'age', 'weight'],
                                 categorical=['group'], groupby=['group'])
        assert (df_groupby['group'] == df_orig['group']).all()
        assert (df_groupby['age'] == df_orig['age']).all()
        assert (df_groupby['weight'] == df_orig['weight']).all()

        # sorted
        df_sorted = self.data_groups.copy()
        table_sorted = TableOne(df_sorted, columns=['group', 'age', 'weight'],
                                categorical=['group'], groupby=['group'],
                                sort=True)
        assert (df_sorted['group'] == df_orig['group']).all()
        assert (df_groupby['age'] == df_orig['age']).all()
        assert (df_groupby['weight'] == df_orig['weight']).all()

        # pval
        df_pval = self.data_groups.copy()
        table_pval = TableOne(df_pval, columns=['group', 'age', 'weight'],
                              categorical=['group'], groupby=['group'],
                              sort=True, pval=True)
        assert (df_pval['group'] == df_orig['group']).all()
        assert (df_groupby['age'] == df_orig['age']).all()
        assert (df_groupby['weight'] == df_orig['weight']).all()

        # pval_adjust
        df_pval_adjust = self.data_groups.copy()
        table_pval_adjust = TableOne(df_pval_adjust,
                                     columns=['group', 'age', 'weight'],
                                     categorical=['group'],
                                     groupby=['group'], sort=True, pval=True,
                                     pval_adjust='bonferroni')
        assert (df_pval_adjust['group'] == df_orig['group']).all()
        assert (df_groupby['age'] == df_orig['age']).all()
        assert (df_groupby['weight'] == df_orig['weight']).all()

        # labels
        df_labels = self.data_groups.copy()
        table_labels = TableOne(df_labels,
                                columns=['group', 'age', 'weight'],
                                categorical=['group'], groupby=['group'],
                                rename={'age': 'age, years'})
        assert (df_labels['group'] == df_orig['group']).all()
        assert (df_groupby['age'] == df_orig['age']).all()
        assert (df_groupby['weight'] == df_orig['weight']).all()

        # limit
        df_limit = self.data_groups.copy()
        table_limit = TableOne(df_limit,
                               columns=['group', 'age', 'weight'],
                               categorical=['group'], groupby=['group'],
                               limit=2)
        assert (df_limit['group'] == df_orig['group']).all()
        assert (df_groupby['age'] == df_orig['age']).all()
        assert (df_groupby['weight'] == df_orig['weight']).all()

        # nonnormal
        df_nonnormal = self.data_groups.copy()
        table_nonnormal = TableOne(df_nonnormal,
                                   columns=['group', 'age', 'weight'],
                                   categorical=['group'], groupby=['group'],
                                   nonnormal=['age'])
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

        table = TableOne(df, columns=columns, groupby=groupby, pval=True,
                         pval_adjust='b')
        tableone_columns = (tableone_columns +
                            list(table.tableone.columns.levels[1]))
        tableone_columns = np.unique(tableone_columns)
        tableone_columns = [c for c in tableone_columns
                            if c not in group_levels]

        for c in tableone_columns:
            # for each output column name in tableone, try them as a group
            df.loc[0:20, 'ICU'] = c
            if 'adjust' in c:
                pval_adjust = 'b'
            else:
                pval_adjust = None

            with assert_raises(InputError):
                table = TableOne(df, columns=columns, groupby=groupby,
                                 pval=True, pval_adjust=pval_adjust)

    @with_setup(setup, teardown)
    def test_label_dictionary_input_pn(self):
        """
        Test columns and rows are relabelled with the label argument
        """
        df = self.data_pn.copy()
        columns = ['Age', 'ICU', 'death']
        categorical = ['death', 'ICU']
        groupby = 'death'

        labels = {'death': 'mortality', 'Age': 'Age, years',
                  'ICU': 'Intensive Care Unit'}

        table = TableOne(df, columns=columns, categorical=categorical,
                         groupby=groupby, rename=labels, label_suffix=False)

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
        table = TableOne(df, columns=columns, label_suffix=False)

        # a call to .index.levels[0] automatically sorts the levels
        # instead, call values and use pd.unique as it preserves order
        tableone_rows = pd.unique([x[0] for x in table.tableone.index.values])

        # default should not sort
        for i, c in enumerate(columns):
            # i+1 because we skip the first row, 'n'
            assert tableone_rows[i+1] == c

        table = TableOne(df, columns=columns, sort=True, label_suffix=False)
        tableone_rows = pd.unique([x[0] for x in table.tableone.index.values])
        for i, c in enumerate(sorted(columns, key=lambda s: s.lower())):
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
            starts_str = "The following continuous column(s) have"
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

        table = TableOne(df, columns=columns, groupby=groupby, pval=True,
                         pval_test_name=True, overall=False)

        assert table.tableone.columns.levels[1][0] == 'Missing'
        assert table.tableone.columns.levels[1][-1] == 'Test'
        assert table.tableone.columns.levels[1][-2] == 'P-Value'

        df.loc[df['death'] == 0, 'death'] = 2

        # without overall column
        table = TableOne(df, columns=columns, groupby=groupby, pval=True,
                         pval_adjust='bonferroni', pval_test_name=True,
                         overall=False)

        assert table.tableone.columns.levels[1][0] == 'Missing'
        assert table.tableone.columns.levels[1][-1] == 'Test'
        assert table.tableone.columns.levels[1][-2] == 'P-Value (adjusted)'

        # with overall column
        table = TableOne(df, columns=columns, groupby=groupby, pval=True,
                         pval_adjust='bonferroni', pval_test_name=True,
                         overall=True)

        assert table.tableone.columns.levels[1][0] == 'Missing'
        assert table.tableone.columns.levels[1][1] == 'Overall'
        assert table.tableone.columns.levels[1][-1] == 'Test'
        assert table.tableone.columns.levels[1][-2] == 'P-Value (adjusted)'

    @with_setup(setup, teardown)
    def test_check_null_counts_are_correct_pn(self):
        """
        Test that the isnull column is correctly reporting number of nulls
        """
        columns = ['Age', 'SysABP', 'Height', 'Weight', 'ICU', 'death']
        categorical = ['ICU', 'death']
        groupby = ['death']

        # test when not grouping
        table = TableOne(self.data_pn, columns=columns,
                         categorical=categorical)

        # get isnull column only
        isnull = table.tableone.iloc[:, 0]
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
        isnull = grouped_table.tableone.iloc[:, 0]
        for i, v in enumerate(isnull):
            # skip empty rows by checking value is not a string
            if 'float' in str(type(v)):
                # check each null count is correct
                col = isnull.index[i][0]
                assert self.data_pn[col].isnull().sum() == v

    # @with_setup(setup, teardown)
    # def test_binary_columns_are_not_converted_to_true_false(self):
    #     """
    #     Fix issue where 0 and 1 were being converted to False and True
    # when set as categorical variables.
    #     """
    #     df = pd.DataFrame({'Feature': [True,True,False,True,False,False,
    #                                    True,False,False,True],
    #         'ID': [1,1,0,0,1,1,0,0,1,0],
    #         'Stuff1': [23,54,45,38,32,59,37,76,32,23],
    #         'Stuff2': [12,12,67,29,24,39,32,65,12,15]})

    #     t = TableOne(df, columns=['Feature','ID'], categorical=['Feature',
    #                                                             'ID'])

    #     # not boolean
    #     assert type(t.tableone.loc['ID'].index[0]) != bool
    #     assert type(t.tableone.loc['ID'].index[1]) != bool

    #     # integer
    #     assert type(t.tableone.loc['ID'].index[0]) == int
    #     assert type(t.tableone.loc['ID'].index[1]) == int

    @with_setup(setup, teardown)
    def test_the_decimals_argument_for_continuous_variables(self):
        """
        For continuous variables, the decimals argument should set the number
        of decimal places for all summary statistics (e.g. mean and standard
        deviation).
        """
        columns = ['Age', 'SysABP', 'Height', 'Weight', 'ICU', 'death']
        categorical = ['ICU', 'death']
        groupby = ['death']
        nonnormal = ['Age']

        # no decimals argument
        # expected result is to default to 1
        t_no_arg = TableOne(self.data_pn, columns=columns,
                            categorical=categorical, groupby=groupby,
                            nonnormal=nonnormal, pval=False,
                            label_suffix=False)

        t_no_arg_group0 = t_no_arg.tableone['Grouped by death'].loc["Weight",
                                                                    "0"].values
        t_no_arg_group0_expected = np.array(['83.0 (23.6)'])

        t_no_arg_group1 = t_no_arg.tableone['Grouped by death'].loc["Weight",
                                                                    "1"].values
        t_no_arg_group1_expected = np.array(['82.3 (25.4)'])

        assert all(t_no_arg_group0 == t_no_arg_group0_expected)
        assert all(t_no_arg_group1 == t_no_arg_group1_expected)

        # decimals = 1
        t1_decimal = TableOne(self.data_pn, columns=columns,
                              categorical=categorical, groupby=groupby,
                              nonnormal=nonnormal, pval=False, decimals=1,
                              label_suffix=False)

        t1_group0 = t1_decimal.tableone['Grouped by death'].loc["Weight",
                                                                "0"].values
        t1_group0_expected = np.array(['83.0 (23.6)'])

        t1_group1 = t1_decimal.tableone['Grouped by death'].loc["Weight",
                                                                "1"].values
        t1_group1_expected = np.array(['82.3 (25.4)'])

        assert all(t1_group0 == t1_group0_expected)
        assert all(t1_group1 == t1_group1_expected)

        # decimals = 2
        t2_decimal = TableOne(self.data_pn, columns=columns,
                              categorical=categorical, groupby=groupby,
                              nonnormal=nonnormal, pval=False, decimals=2,
                              label_suffix=False)

        t2_group0 = t2_decimal.tableone['Grouped by death'].loc["Weight",
                                                                "0"].values
        t2_group0_expected = np.array(['83.04 (23.58)'])

        t2_group1 = t2_decimal.tableone['Grouped by death'].loc["Weight",
                                                                "1"].values
        t2_group1_expected = np.array(['82.29 (25.40)'])

        assert all(t2_group0 == t2_group0_expected)
        assert all(t2_group1 == t2_group1_expected)

        # decimals = {"Age": 0, "Weight":3}
        t3_decimal = TableOne(self.data_pn, columns=columns,
                              categorical=categorical, groupby=groupby,
                              nonnormal=nonnormal, pval=False,
                              decimals={"Age": 0, "Weight": 3},
                              label_suffix=False)

        t3_group0 = t3_decimal.tableone['Grouped by death'].loc["Weight",
                                                                "0"].values
        t3_group0_expected = np.array(['83.041 (23.581)'])

        t3_group1 = t3_decimal.tableone['Grouped by death'].loc["Weight",
                                                                "1"].values
        t3_group1_expected = np.array(['82.286 (25.396)'])

        assert all(t3_group0 == t3_group0_expected)
        assert all(t3_group1 == t3_group1_expected)

    @with_setup(setup, teardown)
    def test_the_decimals_argument_for_categorical_variables(self):
        """
        For categorical variables, the decimals argument should set the number
        of decimal places for the percent only.
        """
        columns = ['Age', 'SysABP', 'Height', 'Weight', 'ICU', 'death']
        categorical = ['ICU', 'death']
        groupby = ['death']
        nonnormal = ['Age']

        # decimals = 1
        t1_decimal = TableOne(self.data_pn, columns=columns,
                              categorical=categorical, groupby=groupby,
                              nonnormal=nonnormal, pval=False, decimals=1,
                              label_suffix=False)

        t1_group0 = t1_decimal.tableone['Grouped by death'].loc["ICU",
                                                                "0"].values
        t1_group0_expected = np.array(['137 (15.9)', '194 (22.5)',
                                      '318 (36.8)', '215 (24.9)'])

        t1_group1 = t1_decimal.tableone['Grouped by death'].loc["ICU",
                                                                "1"].values
        t1_group1_expected = np.array(['25 (18.4)', '8 (5.9)',
                                      '62 (45.6)', '41 (30.1)'])

        assert all(t1_group0 == t1_group0_expected)
        assert all(t1_group1 == t1_group1_expected)

        # decimals = 2
        t2_decimal = TableOne(self.data_pn, columns=columns,
                              categorical=categorical, groupby=groupby,
                              nonnormal=nonnormal, pval=False, decimals=2,
                              label_suffix=False)

        t2_group0 = t2_decimal.tableone['Grouped by death'].loc["ICU",
                                                                "0"].values
        t2_group0_expected = np.array(['137 (15.86)', '194 (22.45)',
                                      '318 (36.81)', '215 (24.88)'])

        t2_group1 = t2_decimal.tableone['Grouped by death'].loc["ICU",
                                                                "1"].values
        t2_group1_expected = np.array(['25 (18.38)', '8 (5.88)',
                                      '62 (45.59)', '41 (30.15)'])

        assert all(t2_group0 == t2_group0_expected)
        assert all(t2_group1 == t2_group1_expected)

        # decimals = {"ICU":3}
        t3_decimal = TableOne(self.data_pn, columns=columns,
                              categorical=categorical, groupby=groupby,
                              nonnormal=nonnormal, pval=False,
                              decimals={"ICU": 3}, label_suffix=False)

        t3_group0 = t3_decimal.tableone['Grouped by death'].loc["ICU",
                                                                "0"].values
        t3_group0_expected = np.array(['137 (15.856)', '194 (22.454)',
                                      '318 (36.806)', '215 (24.884)'])

        t3_group1 = t3_decimal.tableone['Grouped by death'].loc["ICU",
                                                                "1"].values
        t3_group1_expected = np.array(['25 (18.382)', '8 (5.882)',
                                      '62 (45.588)', '41 (30.147)'])

        assert all(t3_group0 == t3_group0_expected)
        assert all(t3_group1 == t3_group1_expected)

        # decimals = {"Age":3}
        # expected result is to default to 1 decimal place
        t4_decimal = TableOne(self.data_pn, columns=columns,
                              categorical=categorical, groupby=groupby,
                              nonnormal=nonnormal, pval=False,
                              decimals={"Age": 3}, label_suffix=False)

        t4_group0 = t4_decimal.tableone['Grouped by death'].loc["ICU",
                                                                "0"].values
        t4_group0_expected = np.array(['137 (15.9)', '194 (22.5)',
                                       '318 (36.8)', '215 (24.9)'])

        t4_group1 = t4_decimal.tableone['Grouped by death'].loc["ICU",
                                                                "1"].values
        t4_group1_expected = np.array(['25 (18.4)', '8 (5.9)',
                                       '62 (45.6)', '41 (30.1)'])

        assert all(t4_group0 == t4_group0_expected)
        assert all(t4_group1 == t4_group1_expected)

    @with_setup(setup, teardown)
    def test_nan_rows_not_deleted_in_categorical_columns(self):
        """
        Test that rows in categorical columns are not deleted if there are null
        values (issue #79).
        """
        # create the dataset
        fruit = [['apple', 'durian', 'pineapple', 'banana'],
                 ['pineapple', 'orange', 'peach', 'lemon'],
                 ['lemon', 'peach', 'lemon', 'banana'],
                 ['durian', 'apple', 'orange', 'lemon'],
                 ['banana', 'durian', 'lemon', 'apple'],
                 ['orange', 'pineapple', 'lemon', 'banana'],
                 ['banana', 'orange', 'apple', 'lemon']]

        df = pd.DataFrame(fruit)
        df.columns = ['basket1', 'basket2', 'basket3', 'basket4']

        # set two of the columns to none
        df.loc[1:3, 'basket2'] = None
        df.loc[2:4, 'basket3'] = None

        # create tableone
        t1 = TableOne(df, label_suffix=False,
                      categorical=['basket1', 'basket2', 'basket3', 'basket4'])

        assert all(t1.tableone.loc['basket1'].index == ['apple', 'banana',
                                                        'durian', 'lemon',
                                                        'orange', 'pineapple'])

        assert all(t1.tableone.loc['basket2'].index == ['durian', 'orange',
                                                        'pineapple'])

        assert all(t1.tableone.loc['basket3'].index == ['apple', 'lemon',
                                                        'peach', 'pineapple'])

        assert all(t1.tableone.loc['basket4'].index == ['apple', 'banana',
                                                        'lemon'])

    @with_setup(setup, teardown)
    def test_pval_correction(self):
        """
        Test the pval_adjust argument
        """
        df = pd.DataFrame({'numbers': [1, 2, 6, 1, 1, 1],
                           'other': [1, 2, 3, 3, 3, 4],
                           'colors': ['red', 'white', 'blue', 'red', 'blue', 'blue'],
                           'even': ['yes', 'no', 'yes', 'yes', 'no', 'yes']})

        t1 = TableOne(df, groupby="even", pval=True, pval_adjust="bonferroni")

        # check the multiplier is correct (3 = no. of reported values)
        pvals_expected = {'numbers, mean (SD)': '1.000',
                          'other, mean (SD)': '1.000',
                          'colors, n (%)': '0.669'}

        group = 'Grouped by even'
        col = 'P-Value (adjusted)'
        for k in pvals_expected:
            assert_equal(t1.tableone.loc[k][group][col].values[0],
                         pvals_expected[k])

        # catch the pval_adjust=True
        with warnings.catch_warnings(record=False) as w:
            warnings.simplefilter('ignore', category=UserWarning)
            t2 = TableOne(df, groupby="even", pval=True, pval_adjust=True)

        for k in pvals_expected:
            assert_equal(t1.tableone.loc[k][group][col].values[0],
                         pvals_expected[k])

    @with_setup(setup, teardown)
    def test_custom_statistical_tests(self):
        """
        Test that the user can specify custom statistical functions.
        """
        # from the example provided at:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html

        # define custom test
        func = mytest

        np.random.seed(12345678)
        n1 = 200
        n2 = 300

        # Baseline distribution
        rvs1 = stats.norm.rvs(size=n1, loc=0., scale=1)
        df1 = pd.DataFrame({'rvs': 'rvs1', 'val': rvs1})

        # Different to rvs1
        # stats.ks_2samp(rvs1, rvs2)
        # (0.20833333333333334, 5.129279597781977e-05)
        rvs2 = stats.norm.rvs(size=n2, loc=0.5, scale=1.5)
        df2 = pd.DataFrame({'rvs': 'rvs2', 'val': rvs2})

        # Similar to rvs1
        # stats.ks_2samp(rvs1, rvs3)
        # (0.10333333333333333, 0.14691437867433876)
        rvs3 = stats.norm.rvs(size=n2, loc=0.01, scale=1.0)
        df3 = pd.DataFrame({'rvs': 'rvs3', 'val': rvs3})

        # Identical to rvs1
        # stats.ks_2samp(rvs1, rvs4)
        # (0.07999999999999996, 0.41126949729859719)
        rvs4 = stats.norm.rvs(size=n2, loc=0.0, scale=1.0)
        df4 = pd.DataFrame({'rvs': 'rvs4', 'val': rvs4})

        # Table 1 for different distributions
        different = df1.append(df2)
        t1_diff = TableOne(data=different, columns=["val"], pval=True,
                           groupby="rvs", htest={"val": func})

        assert_almost_equal(t1_diff._htest_table['P-Value'].val,
                            stats.ks_2samp(rvs1, rvs2)[1])

        # Table 1 for similar distributions
        similar = df1.append(df3)
        t1_similar = TableOne(data=similar, columns=["val"], pval=True,
                              groupby="rvs", htest={"val": func})

        assert_almost_equal(t1_similar._htest_table['P-Value'].val,
                            stats.ks_2samp(rvs1, rvs3)[1])

        # Table 1 for identical distributions
        identical = df1.append(df4)
        t1_identical = TableOne(data=identical, columns=["val"], pval=True,
                                groupby="rvs", htest={"val": func})

        assert_almost_equal(t1_identical._htest_table['P-Value'].val,
                            stats.ks_2samp(rvs1, rvs4)[1])

    @with_setup(setup, teardown)
    def test_compute_standardized_mean_difference_continuous(self):
        """
        Test that pairwise standardized mean difference is computer correctly
        for continuous variables.

        # Ref: Introduction to Meta-Analysis. Michael Borenstein,
        # L. V. Hedges, J. P. T. Higgins and H. R. Rothstein
        # Wiley (2011). Chapter 4. Effect Sizes Based on Means.
        """

        # Example from Hedges 2011:
        # "For example, suppose that a study has sample means X1=103.00,
        #  X2=100.00, sample standard deviations S1=5.5, S2=4.5, and
        #  sample sizes n1=50 and n2=50".

        t = TableOne(pd.DataFrame([1, 2, 3]))

        mean1 = 103.0
        mean2 = 100.0
        n1 = 50
        n2 = 50
        sd1 = 5.5
        sd2 = 4.5

        smd, se = t._cont_smd(mean1=mean1, mean2=mean2, sd1=sd1, sd2=sd2,
                              n1=n1, n2=n2)

        assert_equal(round(smd, 4), -0.5970)
        assert_equal(round(se, 4), 0.2044)

        # Test unbiased estimate using Hedges correction (Hedges, 2011)
        smd, se = t._cont_smd(mean1=mean1, mean2=mean2, sd1=sd1, sd2=sd2,
                              n1=n1, n2=n2, unbiased=True)

        assert_equal(round(smd, 4), -0.5924)
        assert_equal(round(se, 4), 0.2028)

        # Test on input data
        data1 = [1, 2, 3, 4, 5, 6, 7, 8]
        data2 = [2, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        smd_data, se_data = t._cont_smd(data1=data1, data2=data2)

        mean1 = np.mean(data1)
        mean2 = np.mean(data2)
        n1 = len(data1)
        n2 = len(data2)
        sd1 = np.std(data1)
        sd2 = np.std(data2)
        smd_summary, se_summary = t._cont_smd(mean1=mean1, mean2=mean2,
                                              sd1=sd1, sd2=sd2, n1=n1, n2=n2)

        assert_equal(round(smd_data, 4), round(smd_summary, 4))
        assert_equal(round(se_data, 4), round(se_summary, 4))

        # test with the physionet data
        cols = ['Age', 'SysABP', 'Height', 'Weight', 'ICU', 'MechVent', 'LOS',
                'death']
        categorical = ['ICU', 'MechVent', 'death']
        strata = "MechVent"

        t = TableOne(self.data_pn, categorical=categorical, label_suffix=False,
                     groupby=strata, pval=True, pval_test_name=False, smd=True)

        # consistent with R StdDiff() and R tableone
        exp_smd = {'Age': '-0.129',
                   'SysABP': '-0.177',
                   'Height': '-0.073',
                   'Weight': '0.124',
                   'LOS': '0.121'}

        for k in exp_smd:
            smd = t.tableone.loc[k, 'Grouped by MechVent']['SMD (0,1)'][0]
            assert_equal(smd, exp_smd[k])

    @with_setup(setup, teardown)
    def test_compute_standardized_mean_difference_categorical(self):
        """
        Test that pairwise standardized mean difference is computer correctly
        for categorical variables.

        # Ref: Introduction to Meta-Analysis. Michael Borenstein,
        # L. V. Hedges, J. P. T. Higgins and H. R. Rothstein
        # Wiley (2011). Chapter 4. Effect Sizes Based on Means.
        """

        t = TableOne(pd.DataFrame([1, 2, 3]))

        # test with the physionet data
        cols = ['Age', 'SysABP', 'Height', 'Weight', 'ICU', 'MechVent', 'LOS',
                'death']
        categorical = ['ICU', 'MechVent', 'death']
        strata = "MechVent"

        t = TableOne(self.data_pn, categorical=categorical, label_suffix=False,
                     groupby=strata, pval=True, pval_test_name=False, smd=True)

        # consistent with R StdDiff() and R tableone
        exp_smd = {'ICU': '0.747',
                   'MechVent': 'nan',
                   'death': '0.017'}

        for k in exp_smd:
            smd = t.tableone.loc[k, 'Grouped by MechVent']['SMD (0,1)'][0]
            assert_equal(smd, exp_smd[k])
