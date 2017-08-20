# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 11:33:52 2017

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


def visual():
    
     raw_data = pd.read_csv("s_train.data", sep='\t', names=['hour', 'minute', 'second', 'count', 'label', 'elaps']);
     
     #print(raw_data['count'].describe())
     
     raw_data = np.loadtxt("s_train.data", delimiter='\t');
     #data = raw_data[['elaps', 'count']]
     #print(raw_data[:,5]) 
     plt.figure(figsize=(30,20))
     plt.scatter(raw_data[:,5],raw_data[:,3]) 
     p1 = plt.subplot(211)
     p1.plot(raw_data[:,5],raw_data[:,3],"g-",label="count")
     plt.scatter(raw_data[:,5],raw_data[:,3])
     
     raw_data = np.loadtxt("m_train.data", delimiter='\t');
     #print(raw_data[:,4]) 
     p2 = plt.subplot(212)
     p2.plot(raw_data[:,4],raw_data[:,2],"g-",label="count")
     plt.scatter(raw_data[:,4],raw_data[:,2]) 
     plt.show()
    

def test():
    
    raw_data = np.loadtxt("s_train.data", delimiter='\t');
    #data = raw_data[['elaps', 'count']]
    #print(raw_data[:,5]) 
    #plt.scatter(raw_data[:,5],raw_data[:,3]) 
    #p1 = plt.subplot(211)
    #p1.plot(raw_data[:,5],raw_data[:,3],"g-",label="count")
    
    #raw_data = np.loadtxt("m_train.data", delimiter='\t');
    #print(raw_data[:,4]) 
    #p2 = plt.subplot(212)
    #p2.plot(raw_data[:,4],raw_data[:,2],"g-",label="count")
    #plt.scatter(raw_data[:,4],raw_data[:,2]) 
    #plt.show()
    
    
    np.random.seed(1)
    
    xmax = max(raw_data[:,3])
    xmin = min(raw_data[:,3])
    print(xmax, xmin)
    X = raw_data[:,3][:,np.newaxis]
    print(X.shape)
    X_plot = np.linspace(xmin, xmax, 500)[:, np.newaxis]
    #print(X_plot.shape)
    bins = np.linspace(xmin, xmax, 50)
    
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(15,15))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    
    # histogram 1
    ax[0, 0].hist(X[:, 0], bins=bins, fc='#AAAAFF', normed=True)
    ax[0, 0].text(20, 0.18, "Histogram")
    
    # histogram 2
    ax[0, 1].hist(X[:, 0], bins=bins + 0.75, fc='#AAAAFF', normed=True)
    ax[0, 1].text(20, 0.18, "Histogram, bins shifted")
    
    # tophat KDE
    kde = KernelDensity(kernel='tophat', bandwidth=0.75).fit(X)
    log_dens = kde.score_samples(X_plot)
    ax[1, 0].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
    ax[1, 0].text(20, 0.18, "Tophat Kernel Density")
    
    # Gaussian KDE
    kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X)
    log_dens = kde.score_samples(X_plot)
    ax[1, 1].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
    ax[1, 1].text(20, 0.18, "Gaussian Kernel Density")
    
    for axi in ax.ravel():
        axi.plot(X[:, 0], np.zeros(X.shape[0]) - 0.01, '+k')
        axi.set_xlim(xmin, xmax)
        axi.set_ylim(-0.02, 0.2)
    
    for axi in ax[:, 0]:
        axi.set_ylabel('Normalized Density')
    
    for axi in ax[1, :]:
        axi.set_xlabel('x')

def kde(path, column):
    
    raw_data = np.loadtxt(path, delimiter='\t'); 
    np.random.seed(1)
    
    xmax = max(raw_data[:,column])
    xmin = min(raw_data[:,column])
    #print(xmax, xmin)
    X = raw_data[:,column][:,np.newaxis]
    #print(X.shape)
    X_plot = np.linspace(xmin, xmax, 500)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10,10))
    
    for kernel in ['gaussian']:
        kde = KernelDensity(kernel=kernel, bandwidth=0.6).fit(X)
        log_dens = kde.score_samples(X_plot)
        ax.plot(X_plot[:, 0], np.exp(log_dens), 'k.',
                label="kernel = kde_sklearn '{0}'".format(kernel), markersize=2)
    
    ax.text(700, 0.005, "N={0} points".format(X.shape[0]))
    
    ax.legend(loc='upper left')
    ax.plot(X[:, 0],  '+k')
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.001, 0.006)
    plt.show()
     
    ################################
    
    kernel = stats.gaussian_kde(X[:, 0])
    pdf = kernel.evaluate(X_plot[:,0])
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    #ax.imshow((1,1), cmap=plt.cm.gist_earth_r,extent=[xmin, xmax])
    ax.plot(X_plot[:,0], pdf, 'k.', label="kernel = kde_scipy gaussian", markersize=2)
    ax.text(700, 0.0035, "N={0} points".format(X.shape[0]))
    ax.legend(loc='upper left')
    ax.set_xlim([xmin, xmax])
    plt.show()
    
    #print(kernel.factor)
    print(kernel.resample(10))
    
def gmm(path, column):
    
    
    gmm = sklearn.mixture.GaussianMixture(n_components=4)
    # sample data
    #a = np.random.randn(1000)
    # result
    raw_data = np.loadtxt(path, delimiter='\t'); 
    np.random.seed(1)
    
    xmax = max(raw_data[:,column])
    xmin = min(raw_data[:,column])
    #print(xmax, xmin)
    X = raw_data[:,column].reshape(-1,1)
    #print(X.shape)
    X_plot = np.linspace(xmin, xmax, 500).reshape(-1,1)
    
    r = gmm.fit(X)
    
    pdf = r.score_samples(X_plot)
    pdf = np.exp(pdf)
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    #ax.imshow((1,1), cmap=plt.cm.gist_earth_r,extent=[xmin, xmax])
    ax.plot(X_plot[:,0], pdf, 'k.', label="GMM, 4 components", markersize=2)
    ax.text(700, 0.0035, "N={0} points".format(X.shape[0]))
    ax.legend(loc='upper left')
    ax.set_xlim([xmin, xmax])
    plt.show()
    
    
def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    
    """Kernel Density Estimation with Scipy"""
    
    #kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    kde = gaussian_kde(x)
    pdf = kde.evaluate(x_grid)
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    #ax.imshow((1,1), cmap=plt.cm.gist_earth_r,extent=[xmin, xmax])
    ax.plot(x_grid, pdf, 'k.', label="kernel = kde_scipy gaussian", markersize=2)
    ax.text(700, 0.0035, "N={0} points".format(x.shape[0]))
    ax.legend(loc='upper left')
    ax.set_xlim([min(x), max(x)])
    ax.set_ylim(-0.001, 0.006)
    plt.show()


def kde_statsmodels_u(x, x_grid, bandwidth=0.2, **kwargs):
    
    """Univariate Kernel Density Estimation with Statsmodels"""
    
    kde = KDEUnivariate(x)
    #kde.fit(bw=bandwidth, **kwargs)
    kde.fit()
    pdf = kde.evaluate(x_grid)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    #ax.imshow((1,1), cmap=plt.cm.gist_earth_r,extent=[xmin, xmax])
    ax.plot(x_grid, pdf, 'k.', label="kernel = kde_statsmodels_u gaussian", markersize=2)
    ax.text(700, 0.0035, "N={0} points".format(x.shape[0]))
    ax.legend(loc='upper left')
    ax.set_xlim([min(x), max(x)])
    ax.set_ylim(-0.001, 0.006)
    plt.show()


def kde_sklearn(x, x_grid, bandwidth=0.8, **kwargs):
    
    """Kernel Density Estimation with Scikit-learn"""
    
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    #kde_skl = KernelDensity()
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
  
    pdf = np.exp(log_pdf)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    #ax.imshow((1,1), cmap=plt.cm.gist_earth_r,extent=[xmin, xmax])
    ax.plot(x_grid, pdf, '.', label="kernel = kde_sklearn gaussian", markersize=2)
    ax.text(700, 0.0035, "N={0} points".format(x.shape[0]))
    ax.legend(loc='upper left')
    ax.set_xlim([min(x), max(x)])
    ax.set_ylim(-0.001, 0.006)
    plt.show()


def comp_kde(path, column):
    
    
    raw_data = np.loadtxt(path, delimiter='\t'); 
    np.random.seed(1)
    
    xmax = max(raw_data[:,column])
    xmin = min(raw_data[:,column])
    #print(xmax, xmin)
    X = raw_data[:,column]
    #print(X.shape)
    X_plot = np.linspace(xmin, xmax, 500)
    
    
    '''
    fig, ax = plt.subplots(3, 1, sharey=True,
                       figsize=(10, 30))
    
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    
    for i in range(3):
        pdf = kde_funcs[i](X, X_plot)
        ax[i].plot(X_plot, pdf, 'k.', color='blue', alpha=0.3, lw=3)
        ax[i].set_title(kde_funcnames[i])
        ax[i].set_xlim(xmin, xmax)
        ax[i].set_ylim(-0.001, 0.006)
     '''
     
    for i in range(3):
        kde_funcs[i](X, X_plot)

kde_funcs = [kde_statsmodels_u, kde_scipy, kde_sklearn]
kde_funcnames = ['Statsmodels-U', 'Scipy', 'Scikit-learn']
    
if __name__ == '__main__':
    
    comp_kde("s_train.data", 3)
    kde("s_train.data", 3)
    #gmm("s_train.data", 3)
    visual()