TableOne
=========

.. image:: https://travis-ci.org/tompollard/tableone.svg?branch=master
    :target: https://travis-ci.org/tompollard/tableone

tableone is a package for researchers who need to create Table 1, summary
statistics for a patient population. It was inspired by the R package of the
same name by Kazuki Yoshida and Justin Bohn. A demo Jupyter Notebook is
available at: https://github.com/tompollard/tableone/blob/master/tableone.ipynb

Installation
------------

The distribution is hosted on PyPI and directly installable via pip without
needing to clone or download this repository. To install the package from PyPI,
run the following command in your terminal::

    pip install tableone

Example
-------

#. Import libraries::

    from tableone import TableOne
    import pandas as pd

#. Load sample data into a pandas dataframe::

    url="https://raw.githubusercontent.com/tompollard/data/master/primary-biliary-cirrhosis/pbc.csv"
    data=pd.read_csv(url)

#. List of columns containing continuous variables::

    convars = ['age','platelet','ast','bili']

#. List of columns containing categorical variables::

    catvars = ['stage','edema']

#. Optionally, a categorical variable for stratification and a list of non-normal variables::

    strat = 'sex'
    nonnormal = ['bili']

#. Create an instance of TableOne with the input arguments::

    mytable = TableOne(data, convars, catvars, strat, nonnormal)

#. Type the name of the instance in an interpreter::

    mytable

#. ...which prints the following table to screen::

    Stratified by sex
                           f                 m
    ---------------------  ----------------  ----------------
    n                      374               44
    age (mean (std))       55.71 (10.98)     50.16 (10.24)
    platelet (mean (std))  231.14 (85.23)    260.08 (99.42)
    ast (mean (std))       121.99 (47.01)    122.63 (57.92)
    bili (median [IQR])    2.05 [1.30,3.50]  1.30 [0.70,3.40]
    stage (n (%))
    1.0                    18.00 (4.89)      3.00 (6.82)
    2.0                    84.00 (22.83)     8.00 (18.18)
    3.0                    139.00 (37.77)    16.00 (36.36)
    4.0                    127.00 (34.51)    17.00 (38.64)
    edema (n (%))
    0.0                    318.00 (85.03)    36.00 (81.82)
    0.5                    39.00 (10.43)     5.00 (11.36)
    1.0                    17.00 (4.55)      3.00 (6.82)

#. Compute p values by setting the ``pval`` argument to true::

    mytable = TableOne(data, convars, catvars, strat, nonnormal, pval=True)

#. ...which prints::

    Stratified by sex
                           f                 m                 pval    testname
    ---------------------  ----------------  ----------------  ------  --------------
    n                      374               44
    age (mean (std))       55.71 (10.98)     50.16 (10.24)     0.001   One_way_ANOVA
    platelet (mean (std))  231.14 (85.23)    260.08 (99.42)    0.068   One_way_ANOVA
    ast (mean (std))       121.99 (47.01)    122.63 (57.92)    0.949   One_way_ANOVA
    bili (median [IQR])    2.05 [1.30,3.50]  1.30 [0.70,3.40]  0.029   Kruskal-Wallis
    stage (n (%))                                              0.83    Chi-squared
    1.0                    18.00 (4.89)      3.00 (6.82)
    2.0                    84.00 (22.83)     8.00 (18.18)
    3.0                    139.00 (37.77)    16.00 (36.36)
    4.0                    127.00 (34.51)    17.00 (38.64)
    edema (n (%))                                              0.00    Chi-squared
    0.0                    318.00 (85.03)    36.00 (81.82)
    0.5                    39.00 (10.43)     5.00 (11.36)
    1.0                    17.00 (4.55)      3.00 (6.82)


#. Tables can be exported to file in various formats, including LaTeX, Markdown, CSV, and HTML. Files are exported by calling the ``to_format`` methods. For example, mytable can be exported to a CSV named 'mytable.csv' with the following command::

    mytable.to_csv('mytable.csv')
