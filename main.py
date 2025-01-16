# basic libraries
import numpy as np
import pandas as pd
import math
# dataset
from sklearn.datasets import load_iris
#train test splits
from sklearn.model_selection import train_test_split
# accuracy metric
from sklearn.metrics import accuracy_score
# models
from sklearn.neighbors import KNeighborsClassifier      # KNN 
from sklearn.ensemble import AdaBoostClassifier         # AdaBoost
from sklearn.ensemble import GradientBoostingClassifier # Gradient Boosting
from sklearn.naive_bayes import GaussianNB              # Naive Bayes 

#### Extract ####
iris = load_iris()                                              # dataset in sklearn format
iris_pd = pd.DataFrame(iris.data, columns=iris.feature_names)   # iris dataset in pandas dataframe format

#### Data Understanding ####

print(iris_pd.info())

#### Data Preparation ####
x = iris.data
y = iris.target

#### Modeling ####

# we will use the seed = 1234, to always have the same result
seed = 1234

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)

### KNN: ###
# we have to set k to be the number of 
# the appropriate value to k is sqrt(n)/2
k = math.floor(math.sqrt(y.shape[0])/2)
print(k)

# Create KNN classifier object
knn = KNeighborsClassifier(n_neighbors=k)

#training the classifier
knn_applied = knn.fit(x_train, y_train)

correct_results = y_test

#predict the response for the test dataset
model_results_knn = knn_applied.predict(x_test)

# Evaluate the model
accuracy_KNN = accuracy_score(correct_results, model_results_knn)
print("Accuracy of KNN:", accuracy_KNN)

"""
# example of new plant to be classified:
flor_amostra = [[5, 2.1, 1, 0.1]]
flor_np = np.array(flor_amostra)
print(flor_np)
print(iris.target_names[knn.predict(flor_np)]) # setosa
"""

### Adaboost: ###

## Adaboost with Decision Tree (DT) classifier (default) ##

# Create adaboost classifier object
# Adaboost uses DT classifier as default classifier
adaboostDT = AdaBoostClassifier(n_estimators=50,learning_rate=1) #learning_rate--> contributes to the weights to the weak learners.

# training the classifier
adaboostDT_applied = adaboostDT.fit(x_train,y_train)

correct_results = y_test

#predict the response for the test dataset
model_results_adaboostDT = adaboostDT_applied.predict(x_test)

# Evaluate the model
accuracy_adaboostDT = accuracy_score(correct_results, model_results_adaboostDT)
print("Accuracy of Adaboost DT:", accuracy_adaboostDT)

"""
# example of new plant to be classified:
flor_amostra = [[5, 2.1, 1, 0.1]]
flor_np = np.array(flor_amostra)
print(flor_np)
print(iris.target_names[adaboostDT.predict(flor_np)]) # setosa
"""

### Gradient Boosting ###

# Create Gradient Boosting classifier object
gradientboosting = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Train the Gradient Boosting Classifier
gradientboosting_applied = gradientboosting.fit(x_train, y_train)

# Make predictions
model_results_gradientboosting = gradientboosting.predict(x_test)

# Evaluate the model
accuraccuracy_gradientboosting = accuracy_score(correct_results, model_results_gradientboosting)
print("Accuracy of Gradient Boosting:", accuraccuracy_gradientboosting)

### Data Validation and Model Selection ###

### Deploy ###
# To deployment, we'll have 2 functions:
# 1. a function with dataset input and informations about the model winner and mean_accuracy_kfold as output.
# 2. a function with dataset train and dataset test as inputs and species predicted as output.
