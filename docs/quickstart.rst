Quickstart
==========

Install
-------

::

    $ pip install tableone

See :doc:`installation <install>` document for more information.


Run demo
--------

The easiest way to understand what this package does is to evaluate it on data.

The `tableone notebook <https://github.com/tompollard/tableone/blob/master/tableone.ipynb>`_ demonstrates usage of the package. At a high level, you can use the package as follows:

* Import the data into a pandas dataframe
* Run tableone on this dataframe
* Specify your desired output format: text, latex, markdown, etc.

Additional options include:

* specifying a subset of columns
* specifying the data type (categorical, numerical)
* calculating p-values (warning: none of these p-values are adjusted for multiple hypothesis testing)
