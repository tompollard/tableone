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

* Select a subset of columns
* Specify the data type (e.g. `categorical`, `numerical`, `nonnormal`)
* Compute p-values, and adjust for multiple testing (e.g. with the Bonferroni correction)
* Provide a list of alternative labels for variables
* Limit the output of categorical variables to the top N rows.
* Display remarks relating to the appopriateness of summary measures (for example, computing tests for multimodality and normality).
