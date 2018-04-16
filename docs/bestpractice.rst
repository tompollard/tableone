Best Practice 
==============

We recommend seeking guidance from a statistician when using :py:mod:`tableone` for a research study, especially prior to submitting the study for publication. It is beyond the scope of this documentation to provide detailed guidance on summary statistics, but as a primer we provide some considerations for choosing parameters when creating a summary table.

Data visualization
------------------

Plotting the distribution of each variable by group level via histograms, kernel density estimates and boxplots is a crucial component to data analysis pipelines. Vizualisation is often is the only way to detect problematic variables in many real-life scenarios. Some example plots are provided in the `tableone notebook <https://github.com/tompollard/tableone/blob/master/tableone.ipynb>`_.

Normally distributed variables
------------------------------

Variables not listed in the `nonnormal` argument will be summarised by their mean and standard deviation. The mean and standard deviation are often poor estimates of the center or dispersion of a variable’s distribution when the distribution: is asymmetric, has ‘fat’ tails and/or outliers, contains only a very small finite set of values or is multi-modal. Although formal statistical tests are available to detect most of these features, they often are not very useful in small sample sizes [1]_. 

For normally distributed variables, both estimation and hypothesis testing (provided the standard deviations of each group are the same) are more efficient when the variable is not set in the `nonnormal` argument [2]_ [3]_ . This may also hold in some circumstances where the data are clearly not normally distributed, provided the sample sizes are large enough. In other situations, assuming normality when the data is not normally distributed can lead to inefficient or spurious inference.

Non-normally distributed variables
----------------------------------

For numeric variables, including integer and floating point values in addition to some ordered discrete variables, the `nonnormal` argument of TableOne merits some discussion. The practical consequence of including a variable in the `nonnormal` argument is to rely on rank based methods for estimation of the center and variability of the distribution for the relevant variable, along with non-parametric methods to conduct hypothesis testing evaluating if the distributions of all the groups are the same [4]_ [5]_. Median and interquartile range may offer a more robust summary than mean and standard deviation for skewed distributions or in the presence of outliers, but may be misleading in cases such as multimodality.

Comparison of estimates
-----------------------

To supplement data visualization, you may choose to compare two :py:class:`TableOne` tables created from the same dataset: firstly with all numeric variables in the `nonnormal` argument, and subsequently with none of the variables in `nonnormal` argument. Then one can focus on situations where: 

- substantial differences exist between the mean and median estimates
- the median or mean is not well centered between the first and third quartiles [6]_
- large differences exist between the absolute differences in the first and third quartile and the standard deviation, understanding that the interquartile range will be about 35% larger than the standard deviation under normality

A particular situation to note is when the number of groups specified in the `groupby` argument are three or more and the group variances differ to a large degree. Under such a situation it may be preferable to consider the data as non-normal, even if each group’s data were generated from a normal distribution [7]_, particularly when the group sizes are unequal or the sample sizes are small. 

When the number of groups are limited to two, this is addressed using Welch’s two sample t-test which is generally both efficient and robust under unequal variances between two groups [8]_. A similar type of test exists for one-way ANOVA [9]_, but is currently not implemented.

Alternatives to consider
------------------------

Thus far we have suggested methods which vary estimation and hypothesis testing techniques when a normality assumption is not appropriate. Alternatives do exist which may be more practical to your situation. In many circumstances transforming the variable can reduce the influence of asymmetry or other features of the distribution. Under monotone transformations (e.g., logarithm or square root for strictly positive number) this should have little impact on any variable which is included in the `nonnormal` argument, as these methods will typically be invariant to this class of transformation.

Multiple testing
-----------------

If multiple hypotheses are tested, as is commonly the case when numerous variables are summarised in a table, the chance of a rare event increases. As a result, the likelihood of incorrectly rejecting a null hypothesis (i.e., making a Type I error) increases. By default, :py:mod:`tableone` computes the Bonferroni correction to account for multiple testing. This correction addresses the problem of multiple comparisons in a simple way, by dividing the prespecified significance level (Type I error rate, 𝛼) by the number of hypothesis tests conducted. 

The Bonferroni correction is known to over-correct, effectively reducing the statistical power of the tests, particularly when the number of hypotheses are large or when the tests are positively correlated. There are many alternatives which may be more suitable and also widely used, and which should be considered in situations that would be adversely affected by the conservative nature of the Bonferroni correction [10]_ [11]_ [12]_.

Summary
-------

It should be noted that while we have tried to use best practices, automation of even basic statistical tasks can be unsound if done without supervision. We encourage use of :py:mod:`tableone` alongside other methods of descriptive statistics and, in particular, visualization to ensure appropriate data handling.

.. [1] Mohd Razali, Nornadiah & Yap, Bee. (2011). "Power Comparisons of Shapiro-Wilk, 
    Kolmogorov-Smirnov, Lilliefors and Anderson-Darling Tests". Journal of statistical 
    modeling and analytics, volume 2, pp21-33

.. [2] Zimmerman, D. (1987). "Comparative Power of Student T Test and 
    Mann-Whitney U Test for Unequal Sample Sizes and Variances". The Journal of 
    Experimental Education, 55(3), 171-174.

.. [3] Hodges, J., & Lehmann, E. (1956). "The Efficiency of Some Nonparametric 
    Competitors of the t-Test". The Annals of Mathematical Statistics, 27(2), 
    324-335.

.. [4] Lehmann, Erich L and D'Abrera, Howard JM (1975). "Nonparametrics: 
    statistical methods based on ranks". Oxford, England: Holden-Day.

.. [5] Conover, W., & Iman, R. (1981). "Rank Transformations as a Bridge Between 
    Parametric and Nonparametric Statistics". The American Statistician, 35(3), 
    124-129. doi:10.2307/2683975

.. [6] Altman, D., & Bland, J. (1996). "Detecting Skewness From Summary Information. 
    BMJ: British Medical Journal". 313(7066), 1200-1200. 

.. [7] Boneau, C. A. (1960). "The effects of violations of assumptions underlying 
    the t test". Psychological Bulletin, 57(1), 49-64. http://dx.doi.org/10.1037/h0041412

.. [8] Welch Bernard L (1947). "The generalization of ‘Student's’ problem when several 
    different population varlances are involved". Biometrika, Volume 34, Issue 1-2, 
    1 January 1947, Pages 28–35, https://doi.org/10.1093/biomet/34.1-2.28

.. [9] Weerahandi, Samaradasa (1995). “ANOVA under Unequal Error Variances.” 
    Biometrics, vol. 51, no. 2, 1995, pp. 589–599.

.. [10] Benjamini, Yoav & Hochberg, Yosef (1995). "Controlling the false discovery 
    rate: a practical and powerful approach to multiple testing". Journal of the 
    Royal Statistical Society, Series B. 57 (1): 125–133.

.. [11] Holm, S. (1979). "A simple sequentially rejective multiple test procedure". 
    Scandinavian Journal of Statistics. 6 (2): 65–70.

.. [12] Šidák, Z. K. (1967). "Rectangular Confidence Regions for the Means of 
    Multivariate Normal Distributions". Journal of the American Statistical 
    Association. 62 (318): 626–633.