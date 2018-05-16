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

## Goal

I will apply PCA on a wine dataset.

### Importing data
```
wine_original = pd.read_csv('wines.csv')
```
<br/>
<p align="center">
  <img src='images/df_wine.png' width="500">
</p>


Excluding the `red_wine` column

```
wine = wine_original.drop('red_wine', axis=1)
```
### Correlations

```
wc = wine.corr()
```

Since the correlation matrix is symmetric, we can use `np.triu_indices_from` to obtain the indices for the upper-triangle of the matrix only:

```
upper_triangle = np.zeros_like(wc, dtype=np.bool)
upper_triangle[np.triu_indices_from(upper_triangle)] = True
fig, ax = plt.subplots(figsize=(12,4))
ax = sns.heatmap(wc, mask=upper_triangle)
ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=10, rotation=90)
ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=10, rotation=0)
plt.show()
```

### Before applying PCA let us normalize the variables

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
wine_norm = ss.fit_transform(wine)
wine_norm

### Fitting a PCA 

We bulild a `DataFrame` with the PCs adding back `red_wine` column that was dropped.

```
from sklearn.decomposition import PCA
wpca = PCA().fit(wine_norm)
wpcs = wpca.transform(wine_norm)
wpcs = pd.DataFrame(wpcs, columns=['PC'+str(i) for i in range(1, wpcs.shape[1]+1)])
wpcs['red_wine'] = wine_original['red_wine']
```
<br/>
<p align="center">
  <img src='images/df_PC.png' width="500">
</p>

### Plotting the variance explained ratio of the PC

```
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(range(1, wc.shape[1]+1), wpca.explained_variance_ratio_)
ax.scatter(range(1, wine.shape[1]+1), wpca.explained_variance_ratio_, s=200)
ax.set_xlabel('PC')
ax.set_ylabel('Explained Variance')
plt.show()
```
<br/>
<p align="center">
  <img src='images/expl_var.png' width="500">
</p>


### Component weights with corresponding variables for the PCs

```
for i in range(0,4):
    for col, comp in zip(wc.columns, wpca.components_[i]):
        print(col,':',round(comp,3))
    print('')
```

### Seaborn pairplot of PCs


```
sns.pairplot(data=wpcs, vars=['PC1','PC2','PC3'], hue='red_wine', size=2)
```
<br/>
<p align="center">
  <img src='images/hist_PC.png' width="500">
</p>




<br/>
<p align="center">
  <img src='images/corr_pca.png' width="500">
</p>




