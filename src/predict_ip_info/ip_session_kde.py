# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:48:21 2017

@author: Administrator
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import sklearn.mixture
from scipy import stats
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.kernel_density import KDEMultivariate


def kde(path, column):
    
    raw_data = pd.read_csv(path, sep='\t', names=['ip', 'session', 'url'])
    #raw_data = np.loadtxt(path, delimiter='\t'); 
    np.random.seed()

    X = raw_data['session'][:,np.newaxis]
    
    kernel = stats.gaussian_kde(X[:, 0])
    return kernel

def visual(path, column):
    
    raw_data = pd.read_csv(path, sep='\t', names=['ip', 'session', 'url'])
    #raw_data = np.loadtxt(path, delimiter='\t'); 
    np.random.seed(1)
    
    print(raw_data['session'].describe())
    
    xmax = max(raw_data['session'])
    xmin = min(raw_data['session'])
    #print(xmax, xmin)
    X = raw_data['session'][:,np.newaxis]
    #print(X.shape)
    X_plot = np.linspace(xmin, xmax, 500)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10,10))
    
    for kernel in ['gaussian']:
        kde = KernelDensity(kernel=kernel, bandwidth=0.6).fit(X)
        log_dens = kde.score_samples(X_plot)
        ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
                label="kernel = '{0}'".format(kernel), markersize=2)
    
    ax.text(700, 0.005, "N={0} points".format(X.shape[0]))
    
    ax.legend(loc='upper left')
    ax.plot(X[:, 0],  '+k')
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.001, 0.02)
    plt.show()
     
    ################################
    
    kernel = stats.gaussian_kde(X[:, 0])
    pdf = kernel.evaluate(X_plot[:,0])
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    #ax.imshow((1,1), cmap=plt.cm.gist_earth_r,extent=[xmin, xmax])
    ax.plot(X_plot[:,0], pdf, '-', label="kernel = gaussian", markersize=2)
    ax.text(700, 0.0012, "N={0} points".format(X.shape[0]))
    ax.legend(loc='upper left')
    ax.set_xlim([xmin, xmax])
    plt.show()
    

def predict(kernel):
    
    v = int(kernel.resample(1))
    
    #print(v)
    
    if v < 1:
        v = 1
        
    return v
    
if __name__ == '__main__':
     kernel = kde("ip_session.data", 1)
     print("Predict: " + str(predict(kernel)))
     
     visual("ip_session.data", 1)