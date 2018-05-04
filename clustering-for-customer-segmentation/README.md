## Customer Segmentation

In this project I will apply clustering algorithms to the dataset [Wholesale Customers Data Set](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers) from the UCI Machine Learning Repository. 

The dataset contains customers' spending amounts of several product categories .

The data dictionary is:

- FRESH: annual spending (m.u.) on fresh products (Continuous); 
- MILK: annual spending (m.u.) on milk products (Continuous); 
- GROCERY: annual spending (m.u.)on grocery products (Continuous); 
- FROZEN: annual spending (m.u.)on frozen products (Continuous) 
- DETERGENTS_PAPER: annual spending (m.u.) on detergents and paper products (Continuous) 
- DELICATESSEN: annual spending (m.u.)on and delicatessen products (Continuous); 
- CHANNEL: customers' Channel - Horeca (Hotel/Restaurant/Cafe) or Retail channel (Nominal) 
- REGION: customers' Region -  Lisnon, Oporto or Other (Nominal) 

where m.u. stands for monetary units.

## Imports

```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
from IPython.display import display # Allows the use of display() for DataFrames
%matplotlib inline
pd.set_option('display.max_columns', None) # display all columns
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all" # so we can see the value of multiple statements at once.

df = pd.read_csv('customers.csv')
```

## Exploratory Data Analysis

### Cross merchandising

Cross merchandising can be defined as follows ([Ref.1](https://en.wikipedia.org/wiki/Cross_merchandising)):

> The retail practice of marketing or displaying products from different categories together, in order to generate additional revenue for the store, sometimes also known as add-on sales, incremental purchase or secondary product placement. Its main objective is to link different products that complement each other or can logically be used in association. This strategy also aims to improve overall customer experience by allowing them to pick up related goods at the same place instead of having to spend time searching for them.

Consider two categories $A$ and $B$. We can use the dataset to find out the likelihood that customers buying products from $A$ will purchase some proportional quantity from $B$. For that we can use a regression model and set one of the categories in the dataset as target and the remaining ones as features. I will use the following models:
- Linear Regression 
- Decision Tree

I will then compare their `R2`.

```
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

```
Skipping columns Channel and Region:
```
categories = df.columns.tolist()[2:]
```

### Function to run the models and compare them

The functions loops over the model types. It does a train/test split, calculates the score `R2`, determines the coefficients of the linear regression and the features' importance of the Decision Tree Regressor. The feature importance from Decision Trees is defined in the docs as:

> The higher, the more important the feature. The importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance.

```
def cross_categ(category,test_size,models):
    
    categories = df.columns.tolist()[2:] # Skip columns Channel and Region
    
    X,y = df.drop(category,axis=1), df[category]
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=test_size, random_state=0)
    for model in models:
        regr = model().fit(X_train, y_train)
        score = regr.score(X_test, y_test)
        print ("Score for category '{0}' using  {1} model': {2}".format(category,model.__name__, score))
        if model == LinearRegression:
            print ([i for i in categories if i != category])
            print('Coefficients: \n', [round(x,3) for x in regr.coef_][2:])
            print('Intercept is:',regr.intercept_)
            print('Coefficients:',list(zip([i for i in categories if i != category],[round(x,3) for x in regr.coef_][2:])))
            y_pred = regr.predict(X_test)
        elif model == DecisionTreeRegressor:
            print('Feature importances:',list(zip([i for i in categories if i != category],[round(x,3) for x in regr.feature_importances_])))
    return 
```
Calling the function:
```
models = [LinearRegression, DecisionTreeRegressor]
cross_categ('Grocery',0.25,models)
```

The output is:
```
Score for category 'Grocery' using  LinearRegression model': 0.8651525188115992
['Fresh', 'Milk', 'Frozen', 'Detergents_Paper', 'Delicatessen']
Coefficients: 
 [0.023, 0.188, -0.02, 1.651, 0.279]
Intercept is: 818.3234403744482
Coefficients: [('Fresh', 0.023), ('Milk', 0.188), ('Frozen', -0.02), ('Detergents_Paper', 1.651), ('Delicatessen', 0.279)]
Score for category 'Grocery' using  DecisionTreeRegressor model': 0.688660653158595
Feature importances: [('Fresh', 0.0), ('Milk', 0.004), ('Frozen', 0.038), ('Detergents_Paper', 0.064), ('Delicatessen', 0.012)]
```

## To be continued
