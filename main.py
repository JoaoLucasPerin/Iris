# basic libraries
import numpy as np                                              # linear algebra
import pandas as pd                                             # loading data in table form  
import math                                                     # calculus
import seaborn as sns                                           # visualisation 
import matplotlib.pyplot as plt                                 # visualisation
# dataset
from sklearn.datasets import load_iris
#train test splits
from sklearn.model_selection import train_test_split
# accuracy metric
from sklearn.metrics import accuracy_score
# normalization (normal (0,1))
from sklearn.preprocessing import StandardScaler                # dummies
from sklearn.preprocessing import OneHotEncoder                 # dummies
# models
from sklearn.neighbors import KNeighborsClassifier              # KNN 
from sklearn.ensemble import AdaBoostClassifier                 # AdaBoost
from sklearn.ensemble import GradientBoostingClassifier         # Gradient Boosting
from sklearn.naive_bayes import GaussianNB                      # Naive Bayes   
import torch                                                    # Neural Network
import torch.nn as nn                                           # Neural Network
from sklearn.ensemble import RandomForestClassifier             # Random Forest Model
from sklearn.linear_model import LogisticRegression             # Logistic Regression
from sklearn.svm import SVC                                     # Support Vector Machine (SVM)
from xgboost import XGBClassifier                               # XGBoost

#### Options ####
generate_graphs = False
set_seed = True
if set_seed:
    seed = 1234                     # we will use the seed = 1234, to always have the same result
else:
    seed = None

#### Functions ####
# Scatterplots
def scatterplots_iris(data, variables):
    sns.lmplot(
        x = variables[0], # SepalLengthCm
        y = variables[1], # SepalWidthCm
        data = data, 
        fit_reg = False, 
        hue = variables[5], # "Species",
        scatter_kws = {"marker": "D",
                       "s": 50})
    #plt.title('SepalLength vs SepalWidth')

    sns.lmplot(
        x = variables[2], #'PetalLengthCm', 
        y = variables[3], #'PetalWidthCm',
        data = data, 
        fit_reg = False, 
        hue = variables[5], # "Species",
        scatter_kws = {"marker": "D",
                       "s": 50})
    #plt.title('PetalLength vs PetalWidth')

    sns.lmplot(
        x = variables[0], # 'SepalLengthCm', 
        y = variables[2], # 'PetalLengthCm',
        data = data, 
        fit_reg = False, 
        hue = variables[5], # "Species",
        scatter_kws = {"marker": "D",
                       "s": 50})
    #plt.title('SepalLength vs PetalLength')

    sns.lmplot(
        x = variables[1], # 'SepalWidthCm', 
        y = variables[3], # 'PetalWidthCm',
        data = data, 
        fit_reg = False, 
        hue = variables[5], # "Species",
        scatter_kws = {"marker": "D",
                       "s": 50})
    #plt.title('SepalWidth vs PetalWidth')
    plt.show()

def make_adaboost(x_train, y_train):
    ### 1. Adaboost ###

    ## Adaboost with Decision Tree (DT) classifier (default) ##

    # Create adaboost classifier object
    # Adaboost uses DT classifier as default classifier
    adaboostDT = AdaBoostClassifier(n_estimators=50,learning_rate=1) #learning_rate--> contributes to the weights to the weak learners.

    # training the classifier
    adaboostDT_applied = adaboostDT.fit(x_train,y_train)

    return adaboostDT_applied

    """
    # example of new plant to be classified:
    flor_amostra = [[5, 2.1, 1, 0.1]]
    flor_np = np.array(flor_amostra)
    print(flor_np)
    print(iris.target_names[adaboostDT.predict(flor_np)]) # setosa
    """

def make_gradientboosting(x_train, y_train):
    ### 2. Gradient Boosting ###

    # Create Gradient Boosting classifier object
    gradientboosting = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    # Train the Gradient Boosting Classifier
    gradientboosting_applied = gradientboosting.fit(x_train, y_train)

    return gradientboosting_applied

def make_knn(x_train, y_train):
    # we have to set k to be the number of 
    # the appropriate value to k is sqrt(n)/2
    k = math.floor(math.sqrt(y.shape[0])/2)
    print(k)

    # Create KNN classifier object
    knn = KNeighborsClassifier(n_neighbors=k)

    #training the classifier
    knn_applied = knn.fit(x_train, y_train)

    return knn_applied

def make_naivebayes(x_train, y_train):
    ### 4. Naive Bayes ###

    # Create Naive Bayes classifier object
    naivebayes = GaussianNB()

    # Train the Naive Bayes Classifier
    naivebayes_applied = naivebayes.fit(x_train,y_train)

    return naivebayes_applied

def make_neuralnetwork(x_train, y_train):
    # Converting Data for PyTorch
    X_tr_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tr_tensor = torch.tensor(y_train, dtype=torch.long)

    # Building the Neural Network
    class FullyConnected(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(4, 64)  # Input layer to 64 neurons
            self.act1 = nn.ReLU()       # ReLU activation function
            self.l2 = nn.Linear(64, 16) # 64 neurons to 16 neurons
            self.drop = nn.Dropout(0.2) # Dropout for regularization
            self.act2 = nn.ReLU()       # Another ReLU
            self.l3 = nn.Linear(16, 3)  # Output layer to 3 neurons (classes)

        def forward(self, x):
            # Forward pass
            x = self.l1(x)
            x = self.act1(x)
            x = self.l2(x)
            x = self.drop(x)
            x = self.act2(x)
            x = self.l3(x)
            return x

    # Training the Model
    def fit(model):
        epochs = 400
        loss_arr = []
        loss_fn = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=0.002)

        for epoch in range(epochs):
            ypred = model(X_tr_tensor)
            loss = loss_fn(ypred, y_tr_tensor)
            loss_arr.append(loss.item())
            loss.backward()
            optim.step()
            optim.zero_grad()
        plt.plot(loss_arr)
        if generate_graphs:
            plt.show()

    # seting the seed:
    if set_seed:
        torch.manual_seed(seed)

    # Running the Model
    model = FullyConnected()
    #model.to('cuda:0')  # Move model to GPU
    #X_tr_tensor = X_tr_tensor.to('cuda:0')  # Move tensors to GPU
    #y_tr_tensor = y_tr_tensor.to('cuda:0')
    fit(model)

    return model

def make_randomforest(x_train, y_train):
    #Create a Gaussian Classifier
    rf=RandomForestClassifier(n_estimators=100, random_state = seed)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    rf.fit(x_train,y_train)

    return rf

def make_logisticregression(x_train, y_train):
    log = LogisticRegression()
    log.fit(x_train,y_train)

    return log

def make_svm(x_train, y_train):
    np.random.seed(seed) # if None, don't have seed
    svc=SVC()
    svc.fit(x_train, y_train) 
    return svc

def make_xgboost(x_train, y_train):
    xgb = XGBClassifier()
    xgb.fit(x_train, y_train)
    return xgb

#### Extract ####
iris = load_iris()                                              # dataset in sklearn format
iris_pd = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                  columns= iris['feature_names'] + ['target']).astype({'target': int}) \
       .assign(species=lambda x: x['target'].map(dict(enumerate(iris['target_names'])))) # iris dataset in pandas dataframe format

#### Data Understanding ####

print(iris_pd.info())
variables = iris_pd.columns
if generate_graphs:
    scatterplots_iris(iris_pd, variables)

#### Data Preparation ####
x = iris.data
y = iris.target

#### Modeling ####

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)
correct_results = y_test

### 1. Adaboost ###

# Build the model
adaboostDT_applied = make_adaboost(x_train,y_train)

# Make predictions
model_results_adaboostDT = adaboostDT_applied.predict(x_test)

# Evaluate the model
accuracy_adaboostDT = accuracy_score(correct_results, model_results_adaboostDT)
print("Accuracy of Adaboost DT:", accuracy_adaboostDT)

### 2. Gradient Boosting ###

# Build the model
gradientboosting_applied = make_gradientboosting(x_train, y_train)

# Make predictions
model_results_gradientboosting = gradientboosting_applied.predict(x_test)

# Evaluate the model
accuracy_gradientboosting = accuracy_score(correct_results, model_results_gradientboosting)
print("Accuracy of Gradient Boosting:", accuracy_gradientboosting)

### 3. KNN: ###

# Build the model
knn_applied = make_knn(x_train, y_train)

# Make predictions
model_results_knn = knn_applied.predict(x_test)

# Evaluate the model
accuracy_KNN = accuracy_score(correct_results, model_results_knn)
print("Accuracy of KNN:", accuracy_KNN)

### 4. Naive Bayes ###

# Build the model
naivebayes_applied = make_naivebayes(x_train,y_train)

# Make predictions
model_results_naivebayes = naivebayes_applied.predict(x_test)

# Evaluate the model
accuracy_naivebayes = accuracy_score(correct_results, model_results_naivebayes)
print("Accuracy of Naive Bayes:", accuracy_naivebayes)

### 5. Neural Network ###

# Build the model
neuralnetwork_apllied = make_neuralnetwork(x_train, y_train)

# Make predictions
X_ts_tensor = torch.tensor(x_test, dtype=torch.float32)#.to('cuda:0')
ytest_pred = neuralnetwork_apllied(X_ts_tensor)
newytest = torch.argmax(ytest_pred, dim=1)

# Evaluate the Model
accuracy_neuralnetwork = accuracy_score(newytest.cpu(), y_test)
print("Accuracy of Neural Network:", accuracy_neuralnetwork)

"""
# example of new plant to be classified:
flor_amostra = [[5, 2.1, 1, 0.1]]
flor_np = np.array(flor_amostra)
print(flor_np)
y_predicted = torch.argmax(neuralnetwork_apllied(torch.tensor(flor_np, dtype=torch.float32)), dim=1).cpu().detach().numpy()
print(iris.target_names[y_predicted]) # setosa
"""

### 6. Random Forest ###

# Build the model
randomforest_applied = make_randomforest(x_train,y_train)

# Make predictions
model_results_randomforest = randomforest_applied.predict(x_test)

# Evaluate the model
accuracy_randomforest = accuracy_score(correct_results, model_results_randomforest)
print("Accuracy of Random Forest:", accuracy_randomforest)

### 7. Logistic Regression ###

# Build the model
logit_applied = make_logisticregression(x_train, y_train)

# Make predictions
model_results_logit = logit_applied.predict(x_test)

# Evaluate the model
accuracy_logit = accuracy_score(correct_results, model_results_logit)
print("Accuracy of Logistic Regression:", accuracy_logit)

### 8. Support Vector Machine (SVM) ###

# Build the model
svm_applied = make_svm(x_train, y_train)

# Make predictions
model_results_svm = svm_applied.predict(x_test)

# Evaluate the model
accuracy_svm = accuracy_score(correct_results, model_results_svm)
print("Accuracy of SVM:", accuracy_svm)

### 9. XGBoost ###

# Build the model
xgboost_applied = make_xgboost(x_train, y_train)

# Make predictions
model_results_xgboost = xgboost_applied.predict(x_test)

# Evaluate the model
accuracy_xgboost = accuracy_score(correct_results, model_results_xgboost)
print("Accuracy of XGBoost:", accuracy_xgboost)

### Data Validation and Model Selection ###

### Deploy ###
# To deployment, we'll have 2 functions:
# 1. a function with dataset input and informations about the model winner and mean_accuracy_kfold as output.
# 2. a function with dataset train and dataset test as inputs and species predicted as output.
