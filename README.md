# tableone 

tableone is a package for creating "Table 1" summary statistics for a patient 
population. It was inspired by the R package of the same name by Yoshida and 
Bohn.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.837898.svg)](https://doi.org/10.5281/zenodo.837898) [![Documentation Status](https://readthedocs.org/projects/tableone/badge/?version=latest)](https://tableone.readthedocs.io/en/latest/?badge=latest) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/tableone/badges/installer/conda.svg)](https://conda.anaconda.org/conda-forge) [![PyPI version](https://badge.fury.io/py/tableone.svg)](https://badge.fury.io/py/tableone)

## Suggested citation

If you use tableone in your study, please cite the following paper:

> Tom J Pollard, Alistair E W Johnson, Jesse D Raffa, Roger G Mark; tableone: An open source Python package for producing summary statistics for research papers, JAMIA Open, [https://doi.org/10.1093/jamiaopen/ooy012](https://doi.org/10.1093/jamiaopen/ooy012)

## Documentation

For documentation, see: [http://tableone.readthedocs.io/en/latest/](http://tableone.readthedocs.io/en/latest/). An executable demonstration of the package is available [on GitHub](https://github.com/tompollard/tableone/blob/master/tableone.ipynb) as a Jupyter Notebook. The easiest way to try out this notebook is to [open it in Google Colaboratory](https://colab.research.google.com/github/tompollard/tableone/blob/master/tableone.ipynb). A paper describing our motivations for creating the package is available at: [https://doi.org/10.1093/jamiaopen/ooy012](https://doi.org/10.1093/jamiaopen/ooy012).

## A note for users of `tableone`

While we have tried to use best practices in creating this package, automation of even basic statistical tasks can be unsound if done without supervision. We encourage use of `tableone` alongside other methods of descriptive statistics and, in particular, visualization to ensure appropriate data handling. 

It is beyond the scope of our documentation to provide detailed guidance on summary statistics, but as a primer we provide some considerations for choosing parameters when creating a summary table at: [http://tableone.readthedocs.io/en/latest/bestpractice.html](http://tableone.readthedocs.io/en/latest/bestpractice.html). 

*Guidance should be sought from a statistician when using `tableone` for a research study, especially prior to submitting the study for publication*.

## Overview

At a high level, you can use the package as follows:

- Import the data into a pandas DataFrame

![Starting DataFrame ](https://raw.githubusercontent.com/tompollard/tableone/master/docs/images/input_data.png "Starting DataFrame")

- Run tableone on this dataframe to output summary statistics
  
![Table 1](https://raw.githubusercontent.com/tompollard/tableone/master/docs/images/table1.png "Table 1")

- Specify your desired output format: text, latex, markdown, etc.
  
![Export to LaTex](https://raw.githubusercontent.com/tompollard/tableone/master/docs/images/table1_latex.png "Export to LaTex")

Additional options include:

- Select a subset of columns.
- Specify the data type (e.g. `categorical`, `numerical`, `nonnormal`).
- Compute p-values, and adjust for multiple testing (e.g. with the Bonferroni correction).
- Compute standardized mean differences (SMDs).
- Provide a list of alternative labels for variables
- Limit the output of categorical variables to the top N rows.
- Display remarks relating to the appopriateness of summary measures (for example, computing tests for multimodality and normality).

## Installation

To install the package with pip, run:

```pip install tableone```

To install this package with conda, run:
    
```conda install -c conda-forge tableone```

## Example usage

1. Import libraries:

```python
from tableone import TableOne, load_dataset
import pandas as pd
```

2. Load sample data into a pandas dataframe:

```python
data=load_dataset('pn2012')
```

3. Optionally, a list of columns to be included in Table 1:

```python
columns = ['Age', 'SysABP', 'Height', 'Weight', 'ICU', 'death']
```

4. Optionally, a list of columns containing categorical variables:

```python
categorical = ['ICU', 'death']
```

5. Optionally, a categorical variable for stratification, a list of non-normal variables, and a dictionary of alternative labels:

```python
groupby = ['death']
nonnormal = ['Age']
labels={'death': 'mortality'}
```

6. Create an instance of TableOne with the input arguments:

```python
mytable = TableOne(data, columns=columns, categorical=categorical, groupby=groupby, nonnormal=nonnormal, rename=labels, pval=False)
```

7. Display the table using the `tabulate` method. The `tablefmt` argument allows the table to be displayed in multiple formats, including "github", "grid", "fancy_grid", "rst", "html", and "latex".

```python
print(mytable.tabulate(tablefmt = "fancy_grid"))
```

8. ...which prints the following table to screen:

Grouped by mortality:

|           |        | Missing  |        0       |        1       | 
| --------- | ------ | -------- | -------------- | -------------- | 
| n         |        |          | 864            | 136            |
| Age       |        |  0       | 66 [52,78]     | 75 [62,83]     |
| SysABP    |        | 291      | 115.36 (38.34) | 107.57 (49.43) |
| Height    |        | 475      | 170.33 (23.22) | 168.51 (11.31) |
| Weight    |        | 302      | 83.04 (23.58)  | 82.29 (25.40)  | 
| ICU       |  CCU   | 0        | 137 (15.86)    | 25 (18.38)     |
|           |  CSRU  |          | 194 (22.45)    | 8 (5.88)       |  
|           |  MICU  |          | 318 (36.81)    | 62 (45.59)     | 
|           |  SICU  |          | 215 (24.88)    | 41 (30.15)     | 
| mortality |  0     | 0        | 864 (100.0)    |                | 
|           |  1     |          |                | 136 (100.0)    | 

9. Tables can be exported to file in various formats, including LaTeX, CSV, and HTML. Files are exported by calling the ``to_format`` method on the tableone object. For example, mytable can be exported to an Excel spreadsheet named 'mytable.xlsx' with the following command:

```python
mytable.to_excel('mytable.xlsx')
```
