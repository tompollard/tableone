TableOne
=========

.. image:: https://travis-ci.org/tompollard/tableone.svg?branch=master
    :target: https://travis-ci.org/tompollard/tableone

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.837898.svg
    :target: https://doi.org/10.5281/zenodo.837898

.. image:: https://anaconda.org/conda-forge/tableone/badges/installer/conda.svg
    :target: https://conda.anaconda.org/conda-forge

.. image:: https://readthedocs.org/projects/tableone/badge/?version=latest
    :target: http://tableone.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
                

tableone is a package for creating "Table 1" summary statistics for a patient 
population. It was inspired by the R package of the same name by Yoshida and 
Bohn.

Documentation
-------------

For documentation, see: http://tableone.readthedocs.io/en/latest/. An executable demonstration of the package is available as a Jupyter Notebook: https://github.com/tompollard/tableone/blob/master/tableone.ipynb

A note for users of `tableone`
------------------------------

While we have tried to use best practices in creating this package, automation of even basic statistical tasks can be unsound if done without supervision. We encourage use of `tableone` alongside other methods of descriptive statistics and, in particular, visualization to ensure appropriate data handling. 

It is beyond the scope of our documentation to provide detailed guidance on summary statistics, but as a primer we provide some considerations for choosing parameters when creating a summary table at: http://tableone.readthedocs.io/en/latest/bestpractice.html. 

*Guidance should be sought from a statistician when using `tableone` for a research study, especially prior to submitting the study for publication*.

Installation
------------

To install the package with pip, run::

    pip install tableone

To install this package with conda, run::
    
    conda install -c conda-forge tableone

Example
-------

#. Import libraries::

    from tableone import TableOne
    import pandas as pd

#. Load sample data into a pandas dataframe::

    url="https://raw.githubusercontent.com/tompollard/data/master/primary-biliary-cirrhosis/pbc.csv"
    data=pd.read_csv(url)

#. Optionally, a list of columns to be included in Table 1::

    columns = ['age','bili','albumin','ast','platelet','protime',
           'ascites','hepato','spiders','edema','sex', 'trt']

#. Optionally, a list of columns containing categorical variables::

    categorical = ['ascites','hepato','edema','sex','spiders','trt']

#. Optionally, a categorical variable for stratification and a list of non-normal variables::

    groupby = 'trt'
    nonnormal = ['bili']

#. Create an instance of TableOne with the input arguments::

    mytable = TableOne(data, columns, categorical, groupby, nonnormal)

#. Type the name of the instance in an interpreter::

    mytable

#. ...which prints the following table to screen::

    Stratified by trt
                           1.0                2.0                  isnull
    ---------------------  -----------------  -----------------  --------
    n                      158                154                     106
    time (mean (std))      2015.62 (1094.12)  1996.86 (1155.93)         0
    age (mean (std))       51.42 (11.01)      48.58 (9.96)              0
    bili (median [IQR])    1.40 [0.80,3.20]   1.30 [0.72,3.60]          0
    chol (mean (std))      365.01 (209.54)    373.88 (252.48)         134
    albumin (mean (std))   3.52 (0.44)        3.52 (0.40)               0
    copper (mean (std))    97.64 (90.59)      97.65 (80.49)           108
    alk.phos (mean (std))  2021.30 (2183.44)  1943.01 (2101.69)       106
    ast (mean (std))       120.21 (54.52)     124.97 (58.93)          106
    trig (mean (std))      124.14 (71.54)     125.25 (58.52)          136
    platelet (mean (std))  258.75 (100.32)    265.20 (90.73)           11
    protime (mean (std))   10.65 (0.85)       10.80 (1.14)              2
    status (n (%))                                                      0
    0                      83 (52.53)         85 (55.19)
    1                      10 (6.33)          9 (5.84)
    2                      65 (41.14)         60 (38.96)
    ascites (n (%))                                                   106
    0.0                    144 (91.14)        144 (93.51)
    1.0                    14 (8.86)          10 (6.49)
    hepato (n (%))                                                    106
    0.0                    85 (53.80)         67 (43.51)
    1.0                    73 (46.20)         87 (56.49)
    spiders (n (%))                                                   106
    0.0                    113 (71.52)        109 (70.78)
    1.0                    45 (28.48)         45 (29.22)
    edema (n (%))                                                       0
    0.0                    132 (83.54)        131 (85.06)
    0.5                    16 (10.13)         13 (8.44)
    1.0                    10 (6.33)          10 (6.49)
    stage (n (%))                                                       6
    1.0                    12 (7.59)          4 (2.60)
    2.0                    35 (22.15)         32 (20.78)
    3.0                    56 (35.44)         64 (41.56)
    4.0                    55 (34.81)         54 (35.06)
    sex (n (%))                                                         0
    f                      137 (86.71)        139 (90.26)
    m                      21 (13.29)         15 (9.74)    


#. Compute p values by setting the ``pval`` argument to `True`::

    mytable = TableOne(data, columns, categorical, groupby, nonnormal, pval=True)

#. ...which prints::

    Stratified by trt
                           1.0                2.0                  isnull  pval    testname
    ---------------------  -----------------  -----------------  --------  ------  --------------
    n                      158                154                     106
    time (mean (std))      2015.62 (1094.12)  1996.86 (1155.93)         0  0.883   One_way_ANOVA
    age (mean (std))       51.42 (11.01)      48.58 (9.96)              0  0.018   One_way_ANOVA
    bili (median [IQR])    1.40 [0.80,3.20]   1.30 [0.72,3.60]          0  0.842   Kruskal-Wallis
    chol (mean (std))      365.01 (209.54)    373.88 (252.48)         134  0.748   One_way_ANOVA
    albumin (mean (std))   3.52 (0.44)        3.52 (0.40)               0  0.874   One_way_ANOVA
    copper (mean (std))    97.64 (90.59)      97.65 (80.49)           108  0.999   One_way_ANOVA
    alk.phos (mean (std))  2021.30 (2183.44)  1943.01 (2101.69)       106  0.747   One_way_ANOVA
    ast (mean (std))       120.21 (54.52)     124.97 (58.93)          106  0.460   One_way_ANOVA
    trig (mean (std))      124.14 (71.54)     125.25 (58.52)          136  0.886   One_way_ANOVA
    platelet (mean (std))  258.75 (100.32)    265.20 (90.73)           11  0.555   One_way_ANOVA
    protime (mean (std))   10.65 (0.85)       10.80 (1.14)              2  0.197   One_way_ANOVA
    status (n (%))                                                      0  0.894   Chi-squared
    0                      83 (52.53)         85 (55.19)
    1                      10 (6.33)          9 (5.84)
    2                      65 (41.14)         60 (38.96)
    ascites (n (%))                                                   106  0.567   Chi-squared
    0.0                    144 (91.14)        144 (93.51)
    1.0                    14 (8.86)          10 (6.49)
    hepato (n (%))                                                    106  0.088   Chi-squared
    0.0                    85 (53.80)         67 (43.51)
    1.0                    73 (46.20)         87 (56.49)
    spiders (n (%))                                                   106  0.985   Chi-squared
    0.0                    113 (71.52)        109 (70.78)
    1.0                    45 (28.48)         45 (29.22)
    edema (n (%))                                                       0  0.877   Chi-squared
    0.0                    132 (83.54)        131 (85.06)
    0.5                    16 (10.13)         13 (8.44)
    1.0                    10 (6.33)          10 (6.49)
    stage (n (%))                                                       6  0.201   Chi-squared
    1.0                    12 (7.59)          4 (2.60)
    2.0                    35 (22.15)         32 (20.78)
    3.0                    56 (35.44)         64 (41.56)
    4.0                    55 (34.81)         54 (35.06)
    sex (n (%))                                                         0  0.421   Chi-squared
    f                      137 (86.71)        139 (90.26)
    m                      21 (13.29)         15 (9.74)



#. Tables can be exported to file in various formats, including LaTeX, CSV, and HTML. Files are exported by calling the ``to_format`` method on the DataFrame. For example, mytable can be exported to a CSV named 'mytable.csv' with the following command::

    mytable.to_csv('mytable.csv')
