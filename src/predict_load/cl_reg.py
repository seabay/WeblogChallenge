# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:01:14 2017

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVR
from sklearn.svm import SVR

# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    
    
def do_rf(data):
    
    
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)   
    
    X_train = train_set.drop("label", axis=1)
    y_train = train_set['label'].copy();
    print(y_train.shape, X_train.shape)
    
    X_test = test_set.drop("label", axis=1)
    print(X_test.shape)
    y_test = test_set['label'].copy();
    
    rnd_clf = RandomForestClassifier(n_estimators=1500, max_leaf_nodes=16, n_jobs=-1)
    
    '''
    rnd_clf.fit(X_train, y_train)
    y_pred_rf = rnd_clf.predict(X_test)
    n_correct = sum(y_pred_rf == y_test)
    print(n_correct / len(y_pred_rf))
    '''
    
    y_train_pred = cross_val_predict(rnd_clf, X_train, y_train, cv=3)
    #print(y_train_pred)
    n_correct = sum(y_train_pred == y_test)
    print(n_correct / len(y_train_pred))
    conf_mx = confusion_matrix(y_train, y_train_pred)
    #print(conf_mx)
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.show()
    
    
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums
    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    plt.show()
    
    
    
def do_svm(data):
        
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
    
    print(train_set)
    X_train = train_set[:, :132]
    print(X_train.shape)
    y_train = train_set[:, 132];
    print(y_train.shape, X_train.shape)
    
    X_test = test_set[:, :132]
    print(X_test.shape)
    y_test = test_set[:,132];
    
    svm = SVC(kernel="poly", degree=3, coef0=1, C=5)
    svm.fit(X_train, y_train)
    
    y_pred_rf = svm.predict(X_train)
    n_correct = sum(y_pred_rf == y_train)
    print(n_correct / len(y_pred_rf))
    
    y_pred_rf = svm.predict(X_test)
    n_correct = sum(y_pred_rf == y_test)
    print(n_correct / len(y_pred_rf))
 
    
    
def classify():
    
    raw_data = pd.read_csv("s_train.data", sep='\t', names=['hour', 'minute', 'second', 'count', 'label', 'elaps']);
    raw_data['count'].hist(bins=100)
    #raw_data['label'].hist(bins=20)
    
    
    data = raw_data[['hour', 'minute', 'second', 'label']]
    
    do_svm(data)
    
    do_rf(data);
    
    

def do_gbm(data):
    
    print("############ GBM ################")
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
    
    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=80, learning_rate=0.8)
    
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
    
    print("############ SVR ################")
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
    
    svm_reg = SVR(kernel="rbf", gamma=5, C=100, epsilon=20)
    svm_reg.fit(X_train, y_train)
    
    predictions = svm_reg.predict(X_train)
    lin_mse = mean_squared_error(y_train,predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(lin_rmse)
    
    
    X_test = test_set[:, :132]
    y_test = test_set[:,132];
    predictions = svm_reg.predict(X_test)
    lin_mse = mean_squared_error(y_test,predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(lin_rmse)
    
def regress():
    raw_data = pd.read_csv("s_train.data", sep='\t', names=['hour', 'minute', 'second', 'count', 'label', 'elaps']);
    data = raw_data[['hour', 'minute', 'second', 'count']]
    
    print(raw_data["count"].describe())
    corr_matrix = data.corr()
    
    print("######## correlation matrix ##########")
    print(corr_matrix['count'])
    
    do_gbm(data)
    
    do_svr(data)
    
if __name__ == '__main__':
    
    regress()
        