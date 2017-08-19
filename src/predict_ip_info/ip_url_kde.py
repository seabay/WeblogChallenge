# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 17:04:07 2017

@author: Administrator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import sklearn.mixture
from scipy import stats


def kde(path, column):
    
    print('############ KDE #############')
    
    raw_data = pd.read_csv(path, sep='\t', names=['ip', 'session', 'url'])
    #raw_data = np.loadtxt(path, delimiter='\t'); 
    np.random.seed()
    
    data = raw_data.groupby('ip')['url'].sum()
    print(data[0])
    
    xmax = max(data)
    xmin = min(data)
    print(xmax, xmin)
    #X = data[:,np.newaxis]

    kernel = stats.gaussian_kde(data)
    
    return kernel

def visual(path):
    
    print('############ visual #############')
    
    raw_data = pd.read_csv(path, sep='\t', names=['ip', 'session', 'url'])
    data = raw_data.groupby('ip')['url'].sum()
    
    print(data.describe())
    
    #raw_data = np.loadtxt(path, delimiter='\t'); 
    np.random.seed(1)
    
    xmax = max(data)
    xmax = 50
    xmin = min(data)
    #print(xmax, xmin)
    X = data[:,np.newaxis]
    #print(X.shape)
    X_plot = np.linspace(xmin, xmax, 100)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10,10))
    
    for kernel in ['gaussian']:
        kde = KernelDensity(kernel=kernel, bandwidth=0.6).fit(X)
        log_dens = kde.score_samples(X_plot)
        ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
                label="kernel = '{0}'".format(kernel), markersize=2)
    
    ax.text(40, 0.005, "N={0} points".format(X.shape[0]))
    
    ax.legend(loc='upper left')
    ax.plot(X[:, 0],  '+k')
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.001, 0.2)
    plt.show()
     
    ################################
    
    kernel = stats.gaussian_kde(X[:, 0])
    #X_plot = np.linspace(xmin, 200, 100)[:, np.newaxis]
    pdf = kernel.evaluate(X_plot[:,0])
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    #ax.imshow((1,1), cmap=plt.cm.gist_earth_r,extent=[xmin, xmax])
    ax.plot(X_plot[:,0], pdf, '-', label="kernel = gaussian", markersize=2)
    ax.text(40, 0.0012, "N={0} points".format(X.shape[0]))
    ax.legend(loc='upper left')
    ax.set_xlim([xmin, xmax])
    plt.show()

def gmm(path, column):
    
    print('############ GMM #############')
    
    raw_data = pd.read_csv(path, sep='\t', names=['ip', 'session', 'url'])
    data = raw_data.groupby('ip')['url'].sum()
    np.random.seed(1)
    
    xmax = max(data)
    xmax = 50
    xmin = min(data)
    #print(xmax, xmin)
    X = data[:,np.newaxis]
    #print(X.shape)
    X_plot = np.linspace(xmin, xmax, 100).reshape(-1,1)
    #print(X_plot.shape)
    
    #fig = plt.figure(figsize=(10,10))
    #ax = fig.add_subplot(111)
    
    fig, ax = plt.subplots(figsize=(10,10))
    
    ncomponents = [1, 2, 6, 8]
    
    for n in ncomponents:
        
        gmm = sklearn.mixture.GaussianMixture(n_components=n)  
        
        r = gmm.fit(X)
        
        pdf = r.score_samples(X_plot)
        pdf = np.exp(pdf)
        
        #ax.imshow((1,1), cmap=plt.cm.gist_earth_r,extent=[xmin, xmax])
        #ax.plot(X_plot[:,0], pdf, 'k.', label="GMM, 4 components", markersize=2)
        ax.plot(X_plot[:,0], pdf, '-', label="GMM, {0} components".format(n), markersize=2)
        
    ax.text(40, 0.025, "N={0} points".format(X.shape[0]))
    ax.legend(loc='upper left')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim(-0.001, 0.2)
    plt.show()
    ###############################################

def predict(kernel):  
    
    v = np.ceil(np.sum(kernel.resample(10)) / 10)
    print(v)
    
    if v < 1:
        v = 1
        
    return v
    
if __name__ == '__main__':
     #kernel = kde("ip_session.data", 2)
     #print("Predict: " + str(predict(kernel)))
     #gmm("ip_session.data", 2)
     visual("ip_session.data")