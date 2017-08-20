# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:01:14 2017

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
    
def do_rf(data):
    
    print("\n############ RandomForestClassifier ################")
          
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)   
    
    X_train = train_set.drop("label", axis=1)
    y_train = train_set['label'].copy();
    #print(y_train.shape, X_train.shape)
    
    X_test = test_set.drop("label", axis=1)
    #print(X_test.shape)
    y_test = test_set['label'].copy();
    
    '''
    #use GridSearch find best hyperparameters
    param_grid = [
    {'n_estimators': [50, 100, 150, 300, 1000, 1500], 'max_features': [2, 3], 'max_leaf_nodes': [4, 10, 16]},
    {'bootstrap': [False], 'n_estimators': [50, 100, 150, 300, 1000,1500], 'max_features': [2, 3], 'max_leaf_nodes': [4, 10, 16]} ]

    forest_reg = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print(grid_search.best_params_)  # {'n_estimators': 50, 'max_features': 2}
    
    best_model = grid_search.best_estimator_
    '''
    
    best_model = RandomForestClassifier(n_estimators=1500, max_leaf_nodes=16, max_features=3)
    best_model.fit(X_train, y_train)
    
    #y_train_pred = cross_val_predict(best_model, X_train, y_train, cv=10)
    y_train_pred = best_model.predict(X_train)
    n_correct = sum(y_train_pred == y_train)
    print('Train accuracy {0}'.format(n_correct / len(y_train_pred)))
    conf_mx = confusion_matrix(y_train, y_train_pred)
    #print(conf_mx)
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.show()
    
    
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums
    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    plt.show()
    
    
    
    y_pred_rf = best_model.predict(X_test)
    n_correct = sum(y_pred_rf == y_test)
    print('Test accuracy {0}'.format(n_correct / len(y_pred_rf)))
    
    
    
def do_svm(data):
    
    print("\n############ SVM ################")
    
    encoder = LabelBinarizer()
    hour_cat = data["hour"]
    min_cat = data["minute"]
    sds_cat = data["second"]
    hour_cat_1hot = encoder.fit_transform(hour_cat)
    min_cat_1hot = encoder.fit_transform(min_cat)
    sds_cat_1hot = encoder.fit_transform(sds_cat)
    
    data_prep = np.c_[hour_cat_1hot, min_cat_1hot, sds_cat_1hot, data['label']]
    #data_prep = pd.DataFrame(data_prep)
    
    train_set, test_set = train_test_split(data_prep, test_size=0.2, random_state=42)
    
    #print(train_set)
    X_train = train_set[:, :132]
    print(X_train.shape)
    y_train = train_set[:, 132];
    #print(y_train.shape, X_train.shape)
    
    X_test = test_set[:, :132]
    print(X_test.shape)
    y_test = test_set[:,132];
    

    '''
    #use GridSearch find best hyperparameters
    param_grid = [
        {'kernel': ['poly'], 'C': [10., 30., 100., 300., 1000], 'degree':[2,3,4]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100],
         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]

    
    svm = SVC()
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=4)
    grid_search.fit(X_train, y_train)
    
    print(grid_search.best_params_)  # {'gamma': 0.03, 'C': 100, 'kernel': 'rbf'}
    
    best_model = grid_search.best_estimator_
    '''
    
    best_model = SVC(gamma = 0.03, C=100, kernel='rbf')   ### OVO
    best_model.fit(X_train, y_train)
    
    
    #y_train_pred = cross_val_predict(best_model, X_train, y_train, cv=10)
    
    y_train_pred = best_model.predict(X_train)
    n_correct = sum(y_train_pred == y_train)
    print('Train accuracy {0}'.format(n_correct / len(y_train_pred)))
    conf_mx = confusion_matrix(y_train, y_train_pred)
    #print(conf_mx)
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.show()
    
    
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums
    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    plt.show()
    
    
    
    y_pred_rf = best_model.predict(X_test)
    n_correct = sum(y_pred_rf == y_test)
    print('Test accuracy {0}'.format(n_correct / len(y_pred_rf)))
    
    
    
def classify():
    
    raw_data = pd.read_csv("s_train.data", sep='\t', names=['hour', 'minute', 'second', 'count', 'label', 'elaps']);
    #raw_data['count'].hist(bins=100)
    #raw_data['label'].hist(bins=20)
    
    
    data = raw_data[['hour', 'minute', 'second', 'label']]
    
    do_svm(data)
    
    #do_rf(data);
    
    

def do_gbm(data):
    
    print("\n############ GBM ################")
    encoder = LabelBinarizer()
    hour_cat = data["hour"]
    min_cat = data["minute"]
    sds_cat = data["second"]
    hour_cat_1hot = encoder.fit_transform(hour_cat)
    min_cat_1hot = encoder.fit_transform(min_cat)
    sds_cat_1hot = encoder.fit_transform(sds_cat)
    
    data_prep = np.c_[hour_cat_1hot, min_cat_1hot, sds_cat_1hot, data['count']]
    
    train_set, test_set = train_test_split(data_prep, test_size=0.2, random_state=42)
    train_set1, val_set = train_test_split(train_set, test_size=0.2, random_state=42)
    
    X_train = train_set1[:, :132]
    y_train = train_set1[:, 132];
    
    X_val = val_set[:, :132]
    y_val = val_set[:, 132];
   
    X_test = test_set[:, :132]
    y_test = test_set[:,132];
    
    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, learning_rate=0.7)
    
    min_val_error = float("inf")
    error_going_up = 0
    for n_estimators in range(1, 120):
        gbrt.n_estimators = n_estimators
        gbrt.fit(X_train, y_train)
        y_pred = gbrt.predict(X_val)
        val_error = mean_squared_error(y_val, y_pred)
        if val_error < min_val_error:
            min_val_error = val_error
            error_going_up = 0
        else:
            error_going_up += 1
            if error_going_up == 5:
                break # early stopping
    
    predictions = gbrt.predict(X_train)
    lin_mse = mean_squared_error(y_train,predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(lin_rmse)
    
    predictions = gbrt.predict(X_test)
    lin_mse = mean_squared_error(y_test,predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(lin_rmse)
 

def do_svr(data):
    
    print("\n############ SVR ################")
    encoder = LabelBinarizer()
    hour_cat = data["hour"]
    min_cat = data["minute"]
    sds_cat = data["second"]
    hour_cat_1hot = encoder.fit_transform(hour_cat)
    min_cat_1hot = encoder.fit_transform(min_cat)
    sds_cat_1hot = encoder.fit_transform(sds_cat)
    
    data_prep = np.c_[hour_cat_1hot, min_cat_1hot, sds_cat_1hot, data['count']]
    
    train_set, test_set = train_test_split(data_prep, test_size=0.2, random_state=42)
    X_train = train_set[:, :132]
    y_train = train_set[:, 132];
    
    '''
    #use GridSearch find best hyperparameters
    param_grid = [
        {'kernel': ['poly'], 'C': [10., 30., 100., 300., 1000], 'degree':[2,3,4]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100],
         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]

    svm_reg = SVR()
    grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=4)
    grid_search.fit(X_train, y_train)
    
    print(grid_search.best_params_)  # {'gamma': 0.3, 'C': 100, 'kernel': 'rbf'}
    
    best_model = grid_search.best_estimator_
    '''
    
    best_model = SVR(kernel='rbf', C=100, gamma=0.3)
    best_model.fit(X_train, y_train)
    
    predictions = best_model.predict(X_train)
    lin_mse = mean_squared_error(y_train,predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(lin_rmse)
    
    
    X_test = test_set[:, :132]
    y_test = test_set[:,132];
    predictions = best_model.predict(X_test)
    lin_mse = mean_squared_error(y_test,predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(lin_rmse)
    
def do_rfrg(data):
    
    print("\n############ RandomForestRegression ################")
    encoder = LabelBinarizer()
    hour_cat = data["hour"]
    min_cat = data["minute"]
    sds_cat = data["second"]
    hour_cat_1hot = encoder.fit_transform(hour_cat)
    min_cat_1hot = encoder.fit_transform(min_cat)
    sds_cat_1hot = encoder.fit_transform(sds_cat)
    
    data_prep = np.c_[hour_cat_1hot, min_cat_1hot, sds_cat_1hot, data['count']]
    
    train_set, test_set = train_test_split(data_prep, test_size=0.2, random_state=42)
    X_train = train_set[:, :132]
    y_train = train_set[:, 132];
    
    '''
    #use GridSearch find best hyperparameters
    
    param_grid = [
    {'n_estimators': [50, 100, 150, 300, 1000], 'max_features': [2, 3]},
    {'bootstrap': [False], 'n_estimators': [50, 100, 150, 300, 1000], 'max_features': [2, 3]} ]

    forest_reg = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    print(grid_search.best_params_)  # {'n_estimators': 300, 'max_features': 3}
    
    print(grid_search.best_estimator_.max_depth)
    
    best_model = grid_search.best_estimator_
    '''
    
    best_model = RandomForestRegressor(n_estimators=300, max_features=3)
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_train)
    lin_mse = mean_squared_error(y_train,predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(lin_rmse)
    
    
    X_test = test_set[:, :132]
    y_test = test_set[:,132];
    predictions = best_model.predict(X_test)
    lin_mse = mean_squared_error(y_test,predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(lin_rmse)
    
def regression():
    raw_data = pd.read_csv("s_train.data", sep='\t', names=['hour', 'minute', 'second', 'count', 'label', 'elaps']);
    data = raw_data[['hour', 'minute', 'second', 'count']]
    
    '''
    print(raw_data["count"].describe())
    
    corr_matrix = data.corr()
    print("######## correlation matrix ##########")
    print(corr_matrix['count'])
    '''
    
    #do_gbm(data)
    
    #do_svr(data)
    
    do_rfrg(data)
    
if __name__ == '__main__':
    
    regression()
    classify()
        