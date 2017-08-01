import pandas as pd
from tableone import TableOne
from nose.tools import with_setup
import numpy as np

class TestTableOne(object):
    """
    Tests for TableOne
    """

    def setup(self):
        """
        set up test fixtures
        """

        # set random seed
        seed = 12345
        np.random.seed(seed)

        self.create_pbc_dataset()
        self.create_sample_dataset(n = 10000)
        self.create_small_dataset()
        self.create_another_dataset(n = 20)

    def create_pbc_dataset(self):
        """
        create pbc dataset
        """
        url="https://raw.githubusercontent.com/tompollard/data/master/primary-biliary-cirrhosis/pbc.csv"
        self.data_pbc=pd.read_csv(url)

    def create_sample_dataset(self, n):
        """
        create sample dataset
        """
        self.data_sample = pd.DataFrame(index=range(n))

        self.mu, self.sigma = 10, 1
        self.data_sample['normal'] = np.random.normal(self.mu, self.sigma, n)
        self.data_sample['nonnormal'] = np.random.noncentral_chisquare(20,nonc=2,size=n)

        bears = ['Winnie','Paddington','Baloo','Blossom']
        self.data_sample['bear'] = np.random.choice(bears, n, p=[0.5, 0.1, 0.1, 0.3])

        self.data_sample['likeshoney'] = np.nan
        self.data_sample.loc[self.data_sample['bear'] == 'Winnie', 'likeshoney'] = 1
        self.data_sample.loc[self.data_sample['bear'] == 'Baloo', 'likeshoney'] = 1

        self.data_sample['likesmarmalade'] = 0
        self.data_sample.loc[self.data_sample['bear'] == 'Paddington', 'likesmarmalade'] = 1

        self.data_sample['height'] = 0
        self.data_sample.loc[self.data_sample['bear'] == 'Winnie', 'height'] = 6
        self.data_sample.loc[self.data_sample['bear'] == 'Paddington', 'height'] = 4
        self.data_sample.loc[self.data_sample['bear'] == 'Baloo', 'height'] = 20
        self.data_sample.loc[self.data_sample['bear'] == 'Blossom', 'height'] = 7

        self.data_sample['fictional'] = 0
        self.data_sample.loc[self.data_sample['bear'] == 'Winnie', 'fictional'] = 1
        self.data_sample.loc[self.data_sample['bear'] == 'Paddington', 'fictional'] = 1
        self.data_sample.loc[self.data_sample['bear'] == 'Baloo', 'fictional'] = 1
        self.data_sample.loc[self.data_sample['bear'] == 'Blossom', 'fictional'] = 1

    def create_small_dataset(self):
        """
        create small dataset
        """
        self.data_small = pd.DataFrame(index=range(10))
        self.data_small['group1'] = 0
        self.data_small.loc[0:4, 'group1'] = 1
        self.data_small['group2'] = 0
        self.data_small.loc[2:7, 'group2'] = 1
        self.data_small['group3'] = 0
        self.data_small.loc[1:2, 'group3'] = 1
        self.data_small.loc[3:7, 'group3'] = 2

    def create_another_dataset(self, n):
        """
        create another dataset
        """
        self.data_groups = pd.DataFrame(index=range(n))
        self.data_groups['group'] = 'group1'
        self.data_groups.loc[ 2:6, 'group'] = 'group2'
        self.data_groups.loc[ 6:12, 'group'] = 'group3'
        self.data_groups.loc[12: n, 'group'] = 'group4'
        self.data_groups['age'] = range(n)
        self.data_groups['weight'] = [x+100 for x in range(n)]

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
    def test_examples_used_in_the_readme_run_without_raising_error(self):

        columns = ['time','age','bili','chol','albumin','copper',
            'alk.phos','ast','trig','platelet','protime',
            'status', 'ascites', 'hepato', 'spiders', 'edema',
            'stage', 'sex']
        catvars = ['status', 'ascites', 'hepato', 'spiders', 'edema','stage', 'sex']
        strat = 'trt'
        nonnormal = ['bili']
        mytable = TableOne(self.data_pbc, columns, catvars, strat, nonnormal, pval=False)
        mytable = TableOne(self.data_pbc, columns, catvars, strat, nonnormal, pval=True)

    @with_setup(setup, teardown)
    def test_overall_mean_and_std_as_expected_for_cont_variable(self):

        columns=['normal','nonnormal','height']
        table = TableOne(self.data_sample, columns=columns)

        mean =  table._cont_describe['overall'].loc['normal']['mean']
        std = table._cont_describe['overall'].loc['normal']['std']

        assert abs(mean-self.mu) <= 0.02
        assert abs(std-self.sigma) <= 0.02

    @with_setup(setup, teardown)
    def test_overall_n_and_percent_as_expected_for_binary_cat_variable(self):

        categorical=['likesmarmalade']
        table = TableOne(self.data_sample, categorical=categorical)

        notlikefreq = table._cat_describe['overall']['likesmarmalade'][table._cat_describe['overall']['likesmarmalade']['level']==0]['freq'].values[0]
        notlikepercent = table._cat_describe['overall']['likesmarmalade'][table._cat_describe['overall']['likesmarmalade']['level']==0]['percent'].values[0]
        likefreq = table._cat_describe['overall']['likesmarmalade'][table._cat_describe['overall']['likesmarmalade']['level']==1]['freq'].values[0]
        likepercent = table._cat_describe['overall']['likesmarmalade'][table._cat_describe['overall']['likesmarmalade']['level']==1]['percent'].values[0]

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
        table = TableOne(self.data_sample, categorical=categorical)

        likefreq = table._cat_describe['overall']['likeshoney'][table._cat_describe['overall']['likeshoney']['level']==1.0]['freq'].values[0]
        likepercent = table._cat_describe['overall']['likeshoney'][table._cat_describe['overall']['likeshoney']['level']==1.0]['percent'].values[0]

        assert likefreq == 5993
        assert abs(100-likepercent) <= 0.01

    @with_setup(setup, teardown)
    def test_statistical_tests_skipped_if_subgroups_have_zero_observations(self):
        """
        Ensure that the package skips running statistical tests if the subgroups have zero observations
        """
        categorical=['likesmarmalade']
        table = TableOne(self.data_sample, categorical=categorical, strata_col='bear', pval=True)

        assert table._significance_table.loc['likesmarmalade','testname'] == 'Not tested'

    @with_setup(setup, teardown)
    def test_fisher_exact_for_small_cell_count(self):
        """
        Ensure that the package runs Fisher exact if cell counts are <=5 and it's a 2x2
        """
        categorical=['group1','group3']
        table = TableOne(self.data_small, categorical=categorical, strata_col='group2', pval=True)

        # group2 should be tested because it's a 2x2
        # group3 is a 2x3 so should not be tested
        assert table._significance_table.loc['group1','testname'] == 'Fisher exact'
        assert table._significance_table.loc['group3','testname'] == 'Not tested'


    @with_setup(setup, teardown)
    def test_sequence_of_cont_table(self):
        """
        Ensure that the columns align with the values
        """
        columns = ['age','weight']
        categorical = []
        strata_col = 'group'
        t = TableOne(self.data_groups, columns = columns, 
            categorical = categorical, strata_col = strata_col)
        
        # n and weight rows are already ordered, so sorting should not alter the order
        assert t.tableone[0][1:] == sorted(t.tableone[0][1:])
        assert t.tableone[1][1:] == ['0.50 (0.71)', '3.50 (1.29)', '8.50 (1.87)', '15.50 (2.45)']
        assert t.tableone[2][1:] == sorted(t.tableone[2][1:])



