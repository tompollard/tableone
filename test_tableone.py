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

        # load pbc data
        url="https://raw.githubusercontent.com/tompollard/data/master/primary-biliary-cirrhosis/pbc.csv"
        self.data_pbc=pd.read_csv(url)

        # create sample data
        n = 10000
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
    def test_overall_mean_and_std_as_expected_for_cont_variable(self):

        continuous=['normal','nonnormal','height']
        table = TableOne(self.data_sample, continuous=['normal','nonnormal','height'])

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


