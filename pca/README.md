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
  <img src='images/df_wine.png' width="800">
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
<br/>
<p align="center">
  <img src='images/corr_pca.png' width="700">
</p>



### Before applying PCA let us normalize the variables

To [optimize the performance](http://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html#sphx-glr-auto-examples-preprocessing-plot-scaling-importance-py) of the PCA algorithm, the data should be scaled. We will do that using `StandardScaler()` from `sklearn` which standardizes the features onto unit scale.
```
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
wine_norm = ss.fit_transform(wine)
wine_norm
```

<br/>
<p align="center">
  <img src='images/df_scaled.png' width="900">
</p>

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
  <img src='images/df_PC.png' width="800">
</p>

### Plotting the variance explained ratio of the PC

The *explained variance* measures how much information (or variance) can be attributed to each PC. When we reduce the dimensionality, some information, or equivalently, some of the variance is lost. The attribute `explained_variance_ratio_` we find that the first principal component contains 25% of the variance.
```
print('Information contained in each PC:')
PCs = ['PC_{}'.format(i) for i in range(1,1+len(list(wpca.explained_variance_ratio_)))]
EV = [round(i,2) for i in list(wpca.explained_variance_ratio_)]
list(zip(PCs,EV))
```
The output is:
```
Information contained in each PC:
[('PC_1', 0.25),
 ('PC_2', 0.22),
 ('PC_3', 0.14),
 ('PC_4', 0.09),
 ('PC_5', 0.07),
 ('PC_6', 0.06),
 ('PC_7', 0.05),
 ('PC_8', 0.04),
 ('PC_9', 0.04),
 ('PC_10', 0.02),
 ('PC_11', 0.02),
 ('PC_12', 0.0)]

```
Plotting this result:

<br/>
<p align="center">
  <img src='images/expl_var.png' width="500">
</p>


### Component weights with corresponding variables for the PCs
We now print out the weights (eigenvectors) with their corresponding variables. For that we use `.components_`. 

> These are the principal axes in feature space, representing the directions of maximum variance in the data. 

For example:

```
for i in range(0,2):
    for col, comp in zip(wc.columns, wpca.components_[i]):
        print(col,':',round(comp,3))
    print('')
```
The output is:
```
fixed acidity : -0.26
volatile acidity : -0.39
citric acid : 0.15
residual sugar : 0.32
chlorides : -0.31
free sulfur dioxide : 0.42
total sulfur dioxide : 0.47
density : -0.09
pH : -0.21
sulphates : -0.3
alcohol : -0.06
quality : 0.09

fixed acidity : 0.26
volatile acidity : 0.11
citric acid : 0.14
residual sugar : 0.34
chlorides : 0.27
free sulfur dioxide : 0.11
total sulfur dioxide : 0.14
density : 0.55
pH : -0.15
sulphates : 0.12
alcohol : -0.49
quality : -0.3
```

### Red vs white wines

We can check if the first three components are able to differentiate red from white wines using pairplots:

```
sns.pairplot(data=wpcs, vars=['PC1','PC2','PC3'], hue='red_wine', size=2)
```
<br/>
<p align="center">
  <img src='images/hist_PC.png' width="600">
</p>



## To be continued


