#### libraries ####
# basic libraries
import numpy as np                                                      # linear algebra
import pandas as pd                                                     # loading data in table form  
import math                                                             # calculus
import seaborn as sns                                                   # visualisation 
import matplotlib.pyplot as plt                                         # visualisation
# dataset
from sklearn.datasets import load_iris
#train test splits
from sklearn.model_selection import train_test_split
# accuracy metric
from sklearn.metrics import accuracy_score
# models
from sklearn.ensemble import AdaBoostClassifier                         # AdaBoost
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis    # Linear Discriminant Analysis (LDA)
from sklearn.tree import DecisionTreeClassifier                         # Decision Tree Classifier (DTC)
from sklearn.ensemble import GradientBoostingClassifier                 # Gradient Boosting
from sklearn.neighbors import KNeighborsClassifier                      # KNN 
from sklearn.naive_bayes import GaussianNB                              # Naive Bayes   
import torch                                                            # Neural Network - Adam, Adamax, Adamw, Adadelta, Adagrad, NAdam, RMSprop or SGD
import torch.nn as nn                                                   # Neural Network - Adam, Adamax, Adamw, Adadelta, Adagrad, NAdam, RMSprop or SGD
from sklearn.ensemble import RandomForestClassifier                     # Random Forest Model
from sklearn.linear_model import LogisticRegression                     # Logistic Regression (logit)
from sklearn.svm import SVC                                             # Support Vector Machine (SVM)
import xgboost as xgb                                                   # XGBoost
from xgboost import XGBClassifier                                       # XGBoost
from sklearn.model_selection import cross_val_score                     # kfold cross-validation
from sklearn.model_selection import StratifiedKFold                     # kfold cross-validation to XGBoost
import warnings                                                         # supress warnings for deprecated
import time                                                             # script duration

start_time = time.time()

#### Options ####
generate_graphs = False
set_seed = True
if set_seed:
    seed = 1234                     # we will use the seed = 1234, to always have the same result
else:
    seed = None
run_simple_train_test = False
run_kfold = True
warnings.filterwarnings("ignore")

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

### Modeling ###
# Models
# Models without seed aren't changeable
def make_adaboost(x_train, y_train, seed = None):
    ### 1. Adaboost ###

    ## Adaboost with Decision Tree (DT) classifier (default) ##

    # Create adaboost classifier object
    # Adaboost uses DT classifier as default classifier
    adaboostDT = AdaBoostClassifier(n_estimators=50,learning_rate=1,random_state=seed) #learning_rate--> contributes to the weights to the weak learners.

    # training the classifier
    adaboostDT_applied = adaboostDT.fit(x_train,y_train)

    return adaboostDT_applied, adaboostDT

def make_lda(x_train, y_train, seed = None):
    ### 2. Linear Discriminant Analysis (LDA) ###

    lda = LinearDiscriminantAnalysis()
    lda_applied = lda.fit(x_train, y_train)
    return lda_applied, lda

def make_dtc(x_train, y_train, seed = None):
    ### 3. Decision Tree Classifier (DTC) ###

    dtc = DecisionTreeClassifier(random_state = seed)
    dtc_applied = dtc.fit(x_train, y_train)

    return dtc_applied, dtc

def make_gradientboosting(x_train, y_train, seed = None):
    ### 4. Gradient Boosting ###

    # Create Gradient Boosting classifier object
    gradientboosting = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed)

    # Train the Gradient Boosting Classifier
    gradientboosting_applied = gradientboosting.fit(x_train, y_train)

    return gradientboosting_applied, gradientboosting

def make_knn(x_train, y_train):
    ### 5. KNN ###
    # we have to set k to be the number of 
    # the appropriate value to k is sqrt(n)/2
    k = math.floor(math.sqrt(y.shape[0])/2)
    print(k)

    # Create KNN classifier object
    knn = KNeighborsClassifier(n_neighbors=k)

    #training the classifier
    knn_applied = knn.fit(x_train, y_train)

    return knn_applied, knn

def make_naivebayes(x_train, y_train):
    ### 6. Naive Bayes ###

    # Create Naive Bayes classifier object
    naivebayes = GaussianNB()

    # Train the Naive Bayes Classifier
    naivebayes_applied = naivebayes.fit(x_train,y_train)

    return naivebayes_applied, naivebayes

def make_neuralnetwork(x_train, y_train, algo_tx_aprendizado = 'Adam', seed = None):
    ### 7. Neural Network ###
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
        if algo_tx_aprendizado == 'Adam':
            optim = torch.optim.Adam(model.parameters(), lr=0.002) # algo_tx_aprendizado = Adam
        if algo_tx_aprendizado == 'Adamax':
            optim = torch.optim.Adamax(model.parameters()) # algo_tx_aprendizado = Adam
        if algo_tx_aprendizado == 'Adamw':
            optim = torch.optim.AdamW(model.parameters())
        if algo_tx_aprendizado == 'Adadelta':
            optim = torch.optim.Adadelta(model.parameters())
        if algo_tx_aprendizado == 'Adagrad':
            optim = torch.optim.Adagrad(model.parameters())
        if algo_tx_aprendizado == 'Nadam':
            optim = torch.optim.NAdam(model.parameters())
        if algo_tx_aprendizado == 'Rmsprop':
            optim = torch.optim.RMSprop(model.parameters())
        if algo_tx_aprendizado == 'Sgd':
            optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.1) # puting momentum (to reduce oscilaton and speedy up the convergence)     
                                            # and weight_decay (!= 0 => L2 regularization, to prevent overfitting), even with that, SGD is returning bad accuracies to us here.

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
    if set_seed and seed is not None:
        torch.manual_seed(seed)

    # Running the Model
    model = FullyConnected()
    #model.to('cuda:0')  # Move model to GPU
    #X_tr_tensor = X_tr_tensor.to('cuda:0')  # Move tensors to GPU
    #y_tr_tensor = y_tr_tensor.to('cuda:0')
    fit(model)

    return model

def make_randomforest(x_train, y_train, seed = None):
    ### 8. Random Forest ###

    #Create a Gaussian Classifier
    randomforest = RandomForestClassifier(n_estimators=100, random_state = seed)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    randomforest_applied = randomforest.fit(x_train,y_train)

    return randomforest_applied, randomforest

def make_logisticregression(x_train, y_train):
    ### 9. Logistic Regression ###

    logit = LogisticRegression()
    logit_applied = logit.fit(x_train,y_train)

    return logit_applied, logit

def make_svm(x_train, y_train, seed = None):
    ### 10. Support Vector Machine (SVM) ###

    np.random.seed(seed) # if None, don't have seed
    svm=SVC()
    svm_applied = svm.fit(x_train, y_train) 
    return svm_applied, svm

def make_xgboost(x_train, y_train, seed = None):
    ### 11. XGBoost ###

    xgb = XGBClassifier(random_state=seed)
    xgb_applied = xgb.fit(x_train, y_train)
    return xgb, xgb_applied

### Data Validation and Model Selection ###
# kfold
def make_kfold(model = None, x_values = None, y_values = None, k = None, metric = None, is_xgboost = False, is_neuralnetwork = False, algo_tx_aprendizado = 'Adam',seed = None):
    if is_xgboost == True:
        scores = cross_val_score(model,x_values,y_values, scoring=metric, cv=StratifiedKFold())
    if is_neuralnetwork == True:
        x = x_values
        y = y_values

        splits = k
        kf = StratifiedKFold(splits, shuffle=True)
        indices = kf.split(x, y)
        scores = []
        for train, test in indices:
            x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
            
            # Build the model
            if seed is not None:
                neuralnetwork_apllied = make_neuralnetwork(x_train, y_train, algo_tx_aprendizado, seed) # algo_tx_aprendizado default = 'Adam'
            else:
                neuralnetwork_apllied = make_neuralnetwork(x_train, y_train, algo_tx_aprendizado, None) # algo_tx_aprendizado default = 'Adam'

            # Make predictions
            X_ts_tensor = torch.tensor(x_test, dtype=torch.float32)#.to('cuda:0')
            ytest_pred = neuralnetwork_apllied(X_ts_tensor)
            newytest = torch.argmax(ytest_pred, dim=1)

            # Evaluate the Model
            accuracy_neuralnetwork = accuracy_score(newytest.cpu(), y_test)
            scores.append(accuracy_neuralnetwork)
        scores = np.array(scores)
    else: 
        scores = cross_val_score(model, x_values, y_values, cv=k, scoring=metric)

    metric_mean = scores.mean()

    return metric_mean

# function to get model winner
def apply_kfold_and_return_model_winner(
    # mandatory fields:
    x = None, y = None, 
    # optional fields, with k = 5 and metric = 'accuracy' as default values
    k_used = 5, metric_used = 'accuracy',
    # optional fields: list of models possibly passed by the user
    adaboostDT = None, lda = None, dtc = None, gradientboosting = None, knn = None, naivebayes = None, 
    randomforest = None, logit = None, svm = None, xgboost = None, neuralnetwork = None, 
    # optional field: seed
    seed = None, silent = False):

    if silent is False:
        print('Metrics of the models:')
    models_acc_matrix = pd.DataFrame(columns = ['model_name', 'model', 'acc'])

    if adaboostDT is not None:
        metric_adaboost = make_kfold(model = adaboostDT, x_values = x, y_values = y, k = k_used, metric = metric_used) # 1. Adaboost
        if silent is False:
            print(metric_used, 'of Adaboost:', metric_adaboost)
        models_acc_matrix = pd.concat([models_acc_matrix, 
                                       pd.DataFrame({'model_name': ['Adaboost'], 'model': [adaboostDT],'acc': [metric_adaboost]})], ignore_index=True)         

    if lda is not None:
        metric_lda = make_kfold(model = lda, x_values = x, y_values = y, k = k_used, metric = metric_used) # 7. Logistic Regression
        if silent is False:
            print(metric_used, 'of Linear Discriminant Analysis:', metric_lda)
        models_acc_matrix = pd.concat([models_acc_matrix, 
                                       pd.DataFrame({'model_name': ['Linear Discriminant Analysis'], 'model': [lda],'acc': [metric_lda]})], ignore_index=True)                 
    
    if dtc is not None:
        metric_dtc = make_kfold(model = dtc, x_values = x, y_values = y, k = k_used, metric = metric_used) # 7. Logistic Regression
        if silent is False:
            print(metric_used, 'of Decision Tree Classifier:', metric_dtc)
        models_acc_matrix = pd.concat([models_acc_matrix, 
                                       pd.DataFrame({'model_name': ['Decision Tree Classifier'], 'model': [dtc],'acc': [metric_dtc]})], ignore_index=True)                 
     
    if gradientboosting is not None:
        metric_gratientboosting = make_kfold(model = gradientboosting, x_values = x, y_values = y, k = k_used, metric = metric_used) # 2. Gradient Boosting
        if silent is False:
            print(metric_used, 'of Gradient Boosting:', metric_gratientboosting)
        models_acc_matrix = pd.concat([models_acc_matrix, 
                                       pd.DataFrame({'model_name': ['Gradient Boosting'], 'model': [gradientboosting],'acc': [metric_gratientboosting]})], ignore_index=True)         

    if knn is not None:
        metric_knn = make_kfold(model = knn, x_values = x, y_values = y, k = k_used, metric = metric_used) # 3. KNN
        if silent is False:
            print(metric_used, 'of KNN:', metric_knn)
        models_acc_matrix = pd.concat([models_acc_matrix, 
                                       pd.DataFrame({'model_name': ['KNN'],'model': [knn], 'acc': [metric_knn]})], ignore_index=True)         

    if naivebayes is not None:
        metric_naivebayes = make_kfold(model = naivebayes, x_values = x, y_values = y, k = k_used, metric = metric_used) # 4. Naive Bayes
        if silent is False:
            print(metric_used, 'of Naive Bayes:', metric_naivebayes)
        models_acc_matrix = pd.concat([models_acc_matrix, 
                                       pd.DataFrame({'model_name': ['Naive Bayes'], 'model': [naivebayes],'acc': [metric_naivebayes]})], ignore_index=True)         

    if neuralnetwork is not None:
        algo_tx_aprendizado = 'Adam'
        metric_neuralnetwork_adam = make_kfold(model = None, x_values = x, y_values = y, k = k_used, metric = metric_used, 
                                               is_xgboost = False, is_neuralnetwork = True, 
                                               algo_tx_aprendizado = algo_tx_aprendizado, seed = seed) 
        algo_tx_aprendizado = 'Adamax'
        metric_neuralnetwork_adamax = make_kfold(model = None, x_values = x, y_values = y, k = k_used, metric = metric_used, 
                                               is_xgboost = False, is_neuralnetwork = True, 
                                               algo_tx_aprendizado = algo_tx_aprendizado, seed = seed) 
        algo_tx_aprendizado = 'Adamw'
        metric_neuralnetwork_adamw = make_kfold(model = None, x_values = x, y_values = y, k = k_used, metric = metric_used, 
                                               is_xgboost = False, is_neuralnetwork = True, 
                                               algo_tx_aprendizado = algo_tx_aprendizado, seed = seed) 
        algo_tx_aprendizado = 'Adadelta'
        metric_neuralnetwork_adadelta = make_kfold(model = None, x_values = x, y_values = y, k = k_used, metric = metric_used, 
                                               is_xgboost = False, is_neuralnetwork = True, 
                                               algo_tx_aprendizado = algo_tx_aprendizado, seed = seed) 
        algo_tx_aprendizado = 'Adagrad'
        metric_neuralnetwork_adagrad = make_kfold(model = None, x_values = x, y_values = y, k = k_used, metric = metric_used, 
                                               is_xgboost = False, is_neuralnetwork = True, 
                                               algo_tx_aprendizado = algo_tx_aprendizado, seed = seed) 
        algo_tx_aprendizado = 'Nadam'
        metric_neuralnetwork_nadam = make_kfold(model = None, x_values = x, y_values = y, k = k_used, metric = metric_used, 
                                               is_xgboost = False, is_neuralnetwork = True, 
                                               algo_tx_aprendizado = algo_tx_aprendizado, seed = seed) 
        algo_tx_aprendizado = 'Rmsprop'
        metric_neuralnetwork_rmsprop = make_kfold(model = None, x_values = x, y_values = y, k = k_used, metric = metric_used, 
                                               is_xgboost = False, is_neuralnetwork = True, 
                                               algo_tx_aprendizado = algo_tx_aprendizado, seed = seed) 
        algo_tx_aprendizado = 'Sgd'
        metric_neuralnetwork_sgd = make_kfold(model = None, x_values = x, y_values = y, k = k_used, metric = metric_used, 
                                               is_xgboost = False, is_neuralnetwork = True, 
                                               algo_tx_aprendizado = algo_tx_aprendizado, seed = seed) 
        if silent is False:
            print(metric_used, 'of Neural Network - Adam:', metric_neuralnetwork_adam)
            print(metric_used, 'of Neural Network - Adamax:', metric_neuralnetwork_adamax)
            print(metric_used, 'of Neural Network - Adamw:', metric_neuralnetwork_adamw)
            print(metric_used, 'of Neural Network - Adadelta:', metric_neuralnetwork_adadelta)
            print(metric_used, 'of Neural Network - AdaGrad:', metric_neuralnetwork_adagrad)
            print(metric_used, 'of Neural Network - NAdam:', metric_neuralnetwork_nadam)
            print(metric_used, 'of Neural Network - RMSProp:', metric_neuralnetwork_rmsprop)
            print(metric_used, 'of Neural Network - SGD:', metric_neuralnetwork_sgd)
        neuralnetwork_apllied_adam = make_neuralnetwork(x, y, 'Adam', seed)
        neuralnetwork_apllied_adamax = make_neuralnetwork(x, y, 'Adamax', seed)
        neuralnetwork_apllied_adamw = make_neuralnetwork(x, y, 'Adamw', seed)
        neuralnetwork_apllied_adadelta = make_neuralnetwork(x, y, 'Adadelta', seed)
        neuralnetwork_apllied_adagrad = make_neuralnetwork(x, y, 'Adagrad', seed)
        neuralnetwork_apllied_nadam = make_neuralnetwork(x, y, 'Nadam', seed)
        neuralnetwork_apllied_rmsprop = make_neuralnetwork(x, y, 'Rmsprop', seed)
        neuralnetwork_apllied_sgd = make_neuralnetwork(x, y, 'Sgd', seed)
        models_acc_matrix = pd.concat([models_acc_matrix, 
                                       pd.DataFrame({'model_name': ['Neural Network - Adam'], 'model': [neuralnetwork_apllied_adam],
                                                     'acc': [metric_neuralnetwork_adam]})], ignore_index=True)         
        models_acc_matrix = pd.concat([models_acc_matrix, 
                                       pd.DataFrame({'model_name': ['Neural Network - Adamax'], 'model': [neuralnetwork_apllied_adamax],
                                                     'acc': [metric_neuralnetwork_adamax]})], ignore_index=True)     
        models_acc_matrix = pd.concat([models_acc_matrix, 
                                       pd.DataFrame({'model_name': ['Neural Network - Adamw'], 'model': [neuralnetwork_apllied_adamw],
                                                     'acc': [metric_neuralnetwork_adamw]})], ignore_index=True)             
        models_acc_matrix = pd.concat([models_acc_matrix, 
                                       pd.DataFrame({'model_name': ['Neural Network - Adadelta'], 'model': [neuralnetwork_apllied_adadelta],
                                                     'acc': [metric_neuralnetwork_adadelta]})], ignore_index=True)         
        models_acc_matrix = pd.concat([models_acc_matrix, 
                                       pd.DataFrame({'model_name': ['Neural Network - Adagrad'], 'model': [neuralnetwork_apllied_adagrad],
                                                     'acc': [metric_neuralnetwork_adagrad]})], ignore_index=True)         
        models_acc_matrix = pd.concat([models_acc_matrix, 
                                       pd.DataFrame({'model_name': ['Neural Network - NAdam'], 'model': [neuralnetwork_apllied_nadam],
                                                     'acc': [metric_neuralnetwork_nadam]})], ignore_index=True)         
        models_acc_matrix = pd.concat([models_acc_matrix, 
                                       pd.DataFrame({'model_name': ['Neural Network - RMSProp'], 'model': [neuralnetwork_apllied_rmsprop],
                                                     'acc': [metric_neuralnetwork_rmsprop]})], ignore_index=True)         
        models_acc_matrix = pd.concat([models_acc_matrix, 
                                       pd.DataFrame({'model_name': ['Neural Network - SGD'], 'model': [neuralnetwork_apllied_sgd],
                                                     'acc': [metric_neuralnetwork_sgd]})], ignore_index=True)         
        
    if randomforest is not None:
        metric_randomforest = make_kfold(model = randomforest, x_values = x, y_values = y, k = k_used, metric = metric_used) # 6. Random Forest
        if silent is False:
            print(metric_used, 'of Random Forest:', metric_randomforest)
        models_acc_matrix = pd.concat([models_acc_matrix, 
                                       pd.DataFrame({'model_name': ['Random Forest'], 'model': [randomforest],'acc': [metric_randomforest]})], ignore_index=True)         
   
    if logit is not None:
        metric_logisticregression = make_kfold(model = logit, x_values = x, y_values = y, k = k_used, metric = metric_used) # 7. Logistic Regression
        if silent is False:
            print(metric_used, 'of Logistic Regression:', metric_logisticregression)
        models_acc_matrix = pd.concat([models_acc_matrix, 
                                       pd.DataFrame({'model_name': ['Logistic Regression'], 'model': [logit],'acc': [metric_logisticregression]})], ignore_index=True)                 
    
    if svm is not None:
        metric_svm = make_kfold(model = svm, x_values = x, y_values = y, k = k_used, metric = metric_used) # 8. SVM
        if silent is False:
            print(metric_used, 'of SVM:', metric_svm)
        models_acc_matrix = pd.concat([models_acc_matrix, 
                                       pd.DataFrame({'model_name': ['SVM'], 'model': [svm],'acc': [metric_svm]})], ignore_index=True)         

    if xgboost is not None:
        metric_xgboost = make_kfold(model = xgboost, x_values = x, y_values = y, k = k_used, metric = metric_used, is_xgboost=True) # 9. XGBoost
        if silent is False:
            print(metric_used, 'of XGBoost:', metric_xgboost)
        models_acc_matrix = pd.concat([models_acc_matrix, 
                                       pd.DataFrame({'model_name': ['XGBoost'], 'model': [xgboost],'acc': [metric_xgboost]})], ignore_index=True)         
    
    #print(models_acc_matrix)
    model_accs_pd = models_acc_matrix 

    max_metric = max(model_accs_pd['acc'])

    chosen_model = model_accs_pd[ abs(model_accs_pd['acc'] - max_metric) < 0.00001]
    if silent is False:
            print("Biggest accuracy:", max_metric)

    if silent is False:
            print("Model winner:", 
            chosen_model['model_name'].values[0],
            ", with metric=",
            chosen_model['acc'].values[0])

    return chosen_model # returning model informations

# Deploy
# For deployment, we'll have a new function, to get predicted values of new dataset, based on the model winner
def get_predicted_species(model_winner = None, new_x_values = None, dataset_model = None):
    if model_winner is None or new_x_values is None:
        print('missing model, new base or dataset ...')
    else:
        flower_np = np.array(new_x_values)
        if model_winner['model_name'].values[0] in ['Neural Network - Adam', 'Neural Network - Adamax', 'Neural Network - Adamw', 
                                                    'Neural Network - Adadelta', 'Neural Network - Adagrad', 
                                                    'Neural Network - NAdam', 'Neural Network - RMSProp', 'Neural Network - SGD']:
            # In Neural Network algoritm, he have to transform the type of input data...
            # Make predictions
            X_ts_tensor = torch.tensor(flower_np, dtype=torch.float32)#.to('cuda:0')
            y_pred_aux = model_winner['model'].values[0](X_ts_tensor)
            y_pred_aux2 = torch.argmax(y_pred_aux, dim=1).cpu().detach().numpy()
            # print(y_pred_aux2)
            y_predicted = iris.target_names[y_pred_aux2]  
        else:
            y_predicted = iris.target_names[model_winner['model'].values[0].predict(flower_np)]
    return y_predicted

#### Data Extraction ####

iris = load_iris() # dataset in sklearn format
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

if run_simple_train_test:
    #### Applying models with train / test ####

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)
    correct_results = y_test
else:
    x_train = x
    x_test = x 
    y_train = y
    y_test = y 
    correct_results = y

### 1. Adaboost ###

# Build the model
adaboostDT_applied, adaboostDT = make_adaboost(x_train,y_train, seed)

if run_simple_train_test:
    # Make predictions
    model_results_adaboostDT = adaboostDT_applied.predict(x_test)

    # Evaluate the model
    accuracy_adaboostDT = accuracy_score(correct_results, model_results_adaboostDT)
    print("Accuracy of Adaboost DT:", accuracy_adaboostDT)

#### 2. Linear Discriminant Analysis ####
# Build the model
lda_applied, lda = make_lda(x_train,y_train,seed)

if run_simple_train_test:
    # Make predictions
    model_results_lda = lda_applied.predict(x_test)

    # Evaluate the model
    accuracy_lda = accuracy_score(correct_results, model_results_lda)
    print("Accuracy of Linear Discriminant Analysis:", accuracy_lda)

#### 3. Decision Tree Classifier (DTC) ####
# Build the model
dtc_applied, dtc = make_dtc(x_train,y_train,seed)

if run_simple_train_test:
    # Make predictions
    model_results_dtc = dtc_applied.predict(x_test)

    # Evaluate the model
    accuracy_dtc = accuracy_score(correct_results, model_results_dtc)
    print("Accuracy of Decision Tree Classifier:", accuracy_dtc)

### 4. Gradient Boosting ###

# Build the model
gradientboosting_applied, gradientboosting = make_gradientboosting(x_train, y_train, seed)

if run_simple_train_test:
    # Make predictions
    model_results_gradientboosting = gradientboosting_applied.predict(x_test)

    # Evaluate the model
    accuracy_gradientboosting = accuracy_score(correct_results, model_results_gradientboosting)
    print("Accuracy of Gradient Boosting:", accuracy_gradientboosting)

### 5. KNN ###

# Build the model
knn_applied, knn = make_knn(x_train, y_train)

if run_simple_train_test:
    # Make predictions
    model_results_knn = knn_applied.predict(x_test)

    # Evaluate the model
    accuracy_KNN = accuracy_score(correct_results, model_results_knn)
    print("Accuracy of KNN:", accuracy_KNN)

### 6. Naive Bayes ###

# Build the model
naivebayes_applied, naivebayes = make_naivebayes(x_train,y_train)

if run_simple_train_test:
    # Make predictions
    model_results_naivebayes = naivebayes_applied.predict(x_test)

    # Evaluate the model
    accuracy_naivebayes = accuracy_score(correct_results, model_results_naivebayes)
    print("Accuracy of Naive Bayes:", accuracy_naivebayes)

### 7. Neural Network with Adam (Adaptive Moment Estimation) Optimizer ###

# Build the model
neuralnetwork_apllied = make_neuralnetwork(x_train, y_train, 'Adam',seed)

if run_simple_train_test:

    # Make predictions
    X_ts_tensor = torch.tensor(x_test, dtype=torch.float32)#.to('cuda:0')
    ytest_pred = neuralnetwork_apllied(X_ts_tensor)
    newytest = torch.argmax(ytest_pred, dim=1)

    # Evaluate the Model
    accuracy_neuralnetwork = accuracy_score(newytest.cpu(), y_test)
    print("Accuracy of Neural Network with Adam Optimizer:", accuracy_neuralnetwork)

### 7. Neural Network ###
# As Optimizer algorithm, we'll test: 
# 7.1. Adam (Adaptive Moment Estimation with Moving Averages) Optimizer # Default in Neural Networks
neuralnetwork_apllied_adam = make_neuralnetwork(x_train, y_train, 'Adam',seed)
# 7.2. Adamax (Adaptive Moment Estimation with Infinity Norm) Optimizer 
neuralnetwork_apllied_adamax = make_neuralnetwork(x_train, y_train, 'Adamax',seed)
# 7.3. Adamw (Adaptive Moment Estimation with Weight Decay) Optimizer 
neuralnetwork_apllied_adamw = make_neuralnetwork(x_train, y_train, 'Adamw',seed)
# 7.4. Adadelta (Adaptive Gradient Descent Learning Rate with a Moving Window) Optimizer 
neuralnetwork_apllied_adadelta = make_neuralnetwork(x_train, y_train, 'Adadelta', seed)
# 7.5. Adagrad (Adaptive Gradient Descent Learning Rate with Cumulative Sum of Squares) Optimizer 
neuralnetwork_apllied_adagrad = make_neuralnetwork(x_train, y_train, 'Adagrad', seed)
# 7.6. NAdam (Nesterov - Accelerated Adaptative Moment Estimation)
neuralnetwork_apllied_nadam = make_neuralnetwork(x_train, y_train, 'Nadam', seed)
# 7.7. RMSprop (Adaptative Gradient Descent Learning Rate with Root Mean Square Value)
neuralnetwork_apllied_rmsprop = make_neuralnetwork(x_train, y_train, 'Rmsprop', seed)
# 7.8. SGD (Stochastic Gradient Descent)
neuralnetwork_apllied_rmsprop = make_neuralnetwork(x_train, y_train, 'Sgd', seed)
# Build the model

if run_simple_train_test:

    # Make predictions
    X_ts_tensor = torch.tensor(x_test, dtype=torch.float32)#.to('cuda:0')
    ytest_pred = neuralnetwork_apllied(X_ts_tensor)
    newytest = torch.argmax(ytest_pred, dim=1)

    # Evaluate the Model
    accuracy_neuralnetwork = accuracy_score(newytest.cpu(), y_test)
    print("Accuracy of Neural Network with Adam Optimizer:", accuracy_neuralnetwork)

### 8. Random Forest ###

# Build the model
randomforest_applied, randomforest = make_randomforest(x_train,y_train,seed)

if run_simple_train_test:
    # Make predictions
    model_results_randomforest = randomforest_applied.predict(x_test)

    # Evaluate the model
    accuracy_randomforest = accuracy_score(correct_results, model_results_randomforest)
    print("Accuracy of Random Forest:", accuracy_randomforest)

### 9. Logistic Regression ###

# Build the model
logit_applied, logit = make_logisticregression(x_train, y_train)

if run_simple_train_test:
    # Make predictions
    model_results_logit = logit_applied.predict(x_test)

    # Evaluate the model
    accuracy_logit = accuracy_score(correct_results, model_results_logit)
    print("Accuracy of Logistic Regression:", accuracy_logit)

### 10. Support Vector Machine (SVM) ###

# Build the model
svm_applied, svm = make_svm(x_train, y_train, seed)

if run_simple_train_test:

    # Make predictions
    model_results_svm = svm_applied.predict(x_test)

    # Evaluate the model
    accuracy_svm = accuracy_score(correct_results, model_results_svm)
    print("Accuracy of SVM:", accuracy_svm)

### 11. XGBoost ###

# Build the model
xgboost_applied, xgboost = make_xgboost(x_train, y_train, seed)

if run_simple_train_test:
    # Make predictions
    model_results_xgboost = xgboost_applied.predict(x_test)

    # Evaluate the model
    accuracy_xgboost = accuracy_score(correct_results, model_results_xgboost)
    print("Accuracy of XGBoost:", accuracy_xgboost)


# applying examples:

"""
# example of new plant to be classified:
flor_amostra = [[5, 2.1, 1, 0.1]]
flor_np = np.array(flor_amostra)
print(flor_np)
print(iris.target_names[adaboostDT.predict(flor_np)]) # setosa
"""

"""
# example of new plant to be classified:
flor_amostra = [[5, 2.1, 1, 0.1]]
flor_np = np.array(flor_amostra)
print(flor_np)
y_predicted = torch.argmax(neuralnetwork_apllied(torch.tensor(flor_np, dtype=torch.float32)), dim=1).cpu().detach().numpy()
print(iris.target_names[y_predicted]) # setosa
"""

### Data Validation and Model Selection ###

# getting model winner:
metric_used = 'accuracy'
k_used = 5

model_winner = apply_kfold_and_return_model_winner(
    # mandatory fields:
    x = x, y = y, 
    # optional fields, with k = 5 and metric = 'accuracy' as default values
    k_used = k_used, metric_used = metric_used,
    # list of models possibly passed by the user
    adaboostDT = adaboostDT, lda = lda , dtc = dtc, gradientboosting = gradientboosting, knn = knn, naivebayes = naivebayes, 
    randomforest = randomforest, logit = logit, svm = svm, xgboost = xgboost, neuralnetwork = True,
    # seed possibly passed by the user
    seed = seed, silent = False)

### Deploy ###
# For deployment, we'll have a new function, to get predicted values of new dataset, based on the model winner

new_flowers_studied = [[5, 2.1, 1, 0.1], [5.6, 2.8, 4.9, 2.0]] # will be setosa and virginica, in this example

print(get_predicted_species(model_winner = model_winner, 
                            new_x_values = new_flowers_studied, 
                            dataset_model = iris))

print("--- Script performance: %s seconds ---" % (time.time() - start_time))

#####################################################################################################################