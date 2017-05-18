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

Usage
-----

To follow...

Example
-------

1. Import libraries::

    from tableone import TableOne
    import pandas as pd

2. Load sample data into a pandas dataframe::

    url="https://raw.githubusercontent.com/tompollard/data/master/primary-biliary-cirrhosis/pbc.csv"
    data=pd.read_csv(url)

3. List of columns containing continuous variables::

    convars = ['time','age','ascites','hepato','spiders','bili']

4. List of columns containing categorical variables::

    catvars = ['status','edema','stage']

5. Optionally, a categorical variable for stratification and a list of non-normal variables::

    strat = 'trt'
    nonnormal = ['bili']

7. Create an instance of TableOne with the input arguments::

    mytable = TableOne(data, convars, catvars, strat, nonnormal)

8. Type the name of the instance in an interpreter::

    mytable

9. ...which prints the following table to screen::

    Stratified by trt
                          1.0                2.0
    --------------------  -----------------  -----------------
    n                     158                154
    time (mean (std))     2015.62 (1094.12)  1996.86 (1155.93)
    age (mean (std))      51.42 (11.01)      48.58 (9.96)
    ascites (mean (std))  0.09 (0.29)        0.06 (0.25)
    hepato (mean (std))   0.46 (0.50)        0.56 (0.50)
    spiders (mean (std))  0.28 (0.45)        0.29 (0.46)
    bili (median [IQR])   1.40 [0.80,3.20]   1.30 [0.72,3.60]
    status (n (%))
    0                     83.00 (52.53)      85.00 (55.19)
    1                     10.00 (6.33)       9.00 (5.84)
    2                     65.00 (41.14)      60.00 (38.96)
    edema (n (%))
    0.0                   132.00 (83.54)     131.00 (85.06)
    0.5                   16.00 (10.13)      13.00 (8.44)
    1.0                   10.00 (6.33)       10.00 (6.49)
    stage (n (%))
    1.0                   12.00 (7.59)       4.00 (2.60)
    2.0                   35.00 (22.15)      32.00 (20.78)
    3.0                   56.00 (35.44)      64.00 (41.56)
    4.0                   55.00 (34.81)      54.00 (35.06)

9. Tables can be exported to file in various formats, including LaTeX, Markdown, CSV, and HTML. Files are exported by calling the ``to_format`` methods. For example, mytable can be exported to a CSV named 'mytable.csv' with the following command::

    mytable.to_csv('mytable.csv')
