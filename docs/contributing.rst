************************
Contributing to TableOne
************************

We welcome all contributions to the package!

.. contents:: Table of contents:
   :local:


Where to start?
===============

Bug reports, bug fixes, documentation improvements, and other contributions
are welcome. For reporting bugs or suggesting improvements, please use the `GitHub issues
tab <https://github.com/tompollard/tableone/issues/>`_.

Bug reports
===========

Bug reports are core to ensuring the package remains useful for all users.
A complete bug report greatly improves the ability of others to understand and
fix it. For information on how to make a complete bug report, we recommend
you review `this helpful StackOverflow article <https://stackoverflow.com/help/mcve>`_.

Contributing improvements
=========================

Bug fixes or other enhancements are welcome via pull requests. You can `read more
about pull requests on GitHub's website <https://help.github.com/articles/about-pull-requests/>`_.

Contributing to the documentation
=================================

Rewriting small pieces of the documentation as you read through it is a
surefire way of improving them for the next user.

About the documentation
-----------------------

The documentation is written in *reStructuredText*, and subsequently built
using the Python package `Sphinx <http://sphinx.pocoo.org/>`__. The Sphinx
documentation provides `a gentle introduction to
reStructuredText <http://sphinx.pocoo.org/rest.html>`__.

The documentation follows the
`NumPy Docstring Standard <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`__,
which are parsed using the
`napolean extension for sphinx <http://www.sphinx-doc.org/en/1.5.1/ext/napoleon.html>`.

How to build the documentation
------------------------------

Requirements
^^^^^^^^^^^^

To build the documentation you will need to additionally install ``sphinx``.
Furthermore, you'll also need to install the readthedocs theme.
This is easily done using pip::

    pip install sphinx sphinx_rtd_theme

Building the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Navigate to the ``docs`` subfolder and run::

    sphinx-build -b html . _build

Which will build the documentation in the subfolder ``_build``.
Alternatively, you can run the Makefile provided::

    make html
