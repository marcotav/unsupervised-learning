# [PCA](http://nbviewer.jupyter.org/github/marcotav/unsupervised-learning/blob/master/topic-modeling/notebooks/topic-modeling-lda.ipynb) ![image title](https://img.shields.io/badge/python-v3.6-green.svg) ![image title](https://img.shields.io/badge/ntlk-v3.2.5-yellow.svg) ![Image title](https://img.shields.io/badge/sklearn-0.19.1-orange.svg) ![Image title](https://img.shields.io/badge/pandas-0.22.0-red.svg) ![Image title](https://img.shields.io/badge/matplotlib-v2.1.2-orange.svg) ![Image title](https://img.shields.io/badge/gensim-0.3.4-blue.svg)

<p align="center">
  <a href="#intro"> Introduction </a> •
  <a href="#lib"> Libraries </a> •
  <a href="#pro"> Problem domain </a> •
  <a href="#cle"> Cleaning the text </a> •
  <a href="#results"> Results </a> 
</p> 

### Author: [Marco Tavora](http://www.marcotavora.me/)

### Introduction

Principle Component Analysis or PCA is the best known dimensionality reduction algorithm. It combines existing features into fewer ones. Its goals are mainly:
- To transform original features "high-performance" ones
- To reduce the data dimensionality until you are left with the most relevant ones
- Remove multicollinearities

PCA is essentially a coordinate transformation where the original axes are features and the new axes (the new coordinate system for the data) are the *principal components*.

### Example

Let us consider the following example. Suppose my goal is to predict $y$ from the features $x_i$ with $i=1,2,3$. Since this is 3D data it is likely that multicollinearity is present. 

Applying PCA we will obtain "super-predictor variables" called *components*. These are linear combinations of predictors that generate new *principal components* and the latter explain the maximum possible amount of variance in the predictors:

<br/>
<p align="center">
  <img src='images/pca_comp.png' width="300">
</p>
where i=1,2,3. These principal components are uncorrelated. The new axes of principal components are the most concise descriptors of our data. More concretely, each consecutive direction aims at maximizing the remaining variance and each direction is orthogonal to the others

The total variance of your data gets redistributed among the principal components and most variance is captured in the first principal components and the noise is isolated to last principal compoments. Furthermore, there is no covariance between the principal components.


