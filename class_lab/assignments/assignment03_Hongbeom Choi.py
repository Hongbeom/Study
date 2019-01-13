# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 16:05:51 2018

@author: Administrator
"""
# Use the following packages only
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from scipy.misc import comb
from itertools import combinations
import matplotlib.pyplot as plt

# QUESTION 1
def gini(D):
    ######## GINI IMPURITY ########
    # INPUT 
    # D: 1-D array containing different classes of samples
    # OUTPUT    
    # gini: Gini impurity

    # TODO: Gini impurity
    gini = 1
    unique, counts = np.unique(D, return_counts=True)
    for i in counts:
        gini -= np.square(i/len(D))
    return gini

# QUESTION 2
def split_gini(x,y):
    ######## FIND THE BEST SPLIT ########
    # INPUT 
    # x: a input variable 
    # y: a output variable
    # OUTPUT
    # split_point: a scalar number for split. Split is performed based on x>split_point or x<=split_point
    # gain: information gain at the split point
    
    # TODO: find the best split
    IGs = []
    split_point=0    
    gain =0
    for i in x:
        s1 = []
        s2 = []
        for a, b in zip(x, y):
            if a <= i:
                s1.append(b)
            else:
                s2.append(b)
        IGs.append(gini(y)-(gini(s1)*(len(s1)/len(y)))-(gini(s2)*(len(s2)/len(y))))
    gain = max(IGs)
    split_point = x[IGs.index(max(IGs))]
    
    return (split_point,gain)
# QEUSTION 3
def kmeans(X,k,max_iter=300):
    ############ K-MEANS CLUSTERING ##########
    # INPUT
    # X: n by p array (n=# of observations, p=# of input variables)
    # k: the number of clusters
    # max_iter: the maximum number of iteration
    # OUTPUT
    # label: cluster label (len(label)=n)
    # centers: cluster centers (k by p)
    ##########################################
    # If average distance between old centers and new centers is less than 0.000001, stop
    
    # TODO: k-means clustering
    label = []
    centers = []
    itera = 0
    n, p = X.shape
    for i in np.random.randint(0,len(X),k):
        centers.append(X[i])
    centers = np.array(centers)
    while(itera <max_iter):
        label = []
        for i in X:
            tmp = []
            for j in centers:
                tmp.append(np.sqrt(np.sum((i-j)**2)))
            label.append(tmp.index(min(tmp)))
        old_centers = centers
        centers = np.zeros((k,p))
        for idx in range(len(label)):
            centers[label[idx]] += X[idx]
        unique, counts = np.unique(label, return_counts=True)
        for idx in range(len(centers)):
            centers[idx] /= counts[idx]
        sum_distance = 0
        for a, b in zip(old_centers, centers):
            sum_distance +=  np.sqrt(np.sum((a-b)**2))
        if(sum_distance/k < 0.000001):
            break;
        itera += 1
    
    return (label, centers)

# QUESTION 4
def cal_support(data,rule):
    ######## CALCULATE SUPPORT OF ASSOCIATION RULE ########
    # INPUT
    # data: transaction data, each row contains items
    # rule: array or list with two elements, rule[0] is a set of condition items and rule[1] is a set of result itmes
    # OUTPUT
    # support: support of the rule
    #######################################################
    
    # TODO: support 
    support = 0
    up = 0
    for i in data:
        check = []
        for j in rule[0]:
            if j in i:
                check.append(True)
            else:
                check.append(False)
        for k in rule[1]:
            if k in i:
                check.append(True)
            else:
                check.append(False)
        if False not in check:
            up += 1
    support = up/len(data)
    return support

# QUESTION 5
def cal_conf(data,rule):
    ######## CALCULATE CONFIDENCE OF ASSOCIATION RULE ########
    # INPUT
    # data: transaction data, each row contains items
    # rule: array or list with two elements, rule[0] is a set of condition items and rule[1] is a set of result itmes
    # OUTPUT
    # confidence: confidence of the rule
    #########################################################
    
    # TODO: confidence
    confidence = 0
    down = 0
    for i in data:
        check = []
        for j in rule[0]:
            if j in i:
                check.append(True)
            else:
                check.append(False)
        if False not in check:
            down += 1
    confidence = cal_support(data, rule)/(down/len(data))
    return confidence

# QEUSTION 6
def generate_ck(data,k,Lprev=[]):
    ######## GENERATE Ck ########
    # INPUT
    # data: transaction data, each row contains items
    # k: the number of items in sets
    # Lprev: L(k-1) for k>=2
    # OUTPUT
    # Ck: candidates of frequent items sets with k items
    ##############################
    
    # TODO: Ck
    if k==1:
        Ck=[]
        for i in data:
            for j in i:
                if [j] not in Ck:
                    Ck.append([j])
        return Ck
    else:
        Ck=[]
        tmp = list(combinations(list(set(sum(Lprev, []))),k)) 
        for i in tmp:
            i = list(i)
            tmp2 = list(combinations(i, k-1))
            check = []
            for j in tmp2:
                j = list(j)
                if j in Lprev:
                    check.append(True)
                else:
                    check.append(False)
            if False not in check:
                Ck.append(i)
        return Ck

# QEUSTION 7
def generate_lk(data,Ck,min_sup):
    ######## GENERATE Lk ########
    # INPUT
    # data: transaction data, each row contains items
    # Ck: candidates of frequent items sets with k items
    # min_sup: minimum support
    # OUTPUT
    # Lk: frequent items sets with k items
    ##############################
    
    # TODO: Lk
    # Use cal_support
    Lk=[]
    for i in Ck:
        if cal_support(data, [i,[]]) >= min_sup:
            Lk.append(i)            
    return Lk

# QEUSTION 8
def PCA(X,k):
    ######## PCA ########
    # INPUT
    # X: n by p array (n=# of observations, p=# of input variables)
    # k: the number of components
    # OUTPUT
    # components: p by k array, each column corresponds to PC in order. (the first PC is the first column)
    
    # TODO: PCA
    # Hint: use numpy.linalg.eigh
    components = []
    n,p = X.shape
    stX = X -X.mean(0)
    cov = np.matmul(stX.T, stX)/(n)
    eigenValue, eigenVector = np.linalg.eigh(cov, 'U')
    eigenValue = list(eigenValue)
    for i in range(1, k+1):
        components.append(eigenVector[:,-i])
    components = np.array(components).T
    return components
    
if __name__=='main':    
    cancer=pd.read_csv('https://drive.google.com/uc?export=download&id=1-83EtpdXI_bNWlWD7v-t_7XLJgnwocxg')
    iris=load_iris()
    trans=pd.read_csv('https://drive.google.com/uc?export=download&id=1F_6wOpWqO-yXfbpfSCXfX6_uV4YhOPqD', index_col=0)
    trans=[x.split(',') for x in trans['Items'].values]
    #################### TEST YOUR CODE ####################
    
    #Problem 2
    variable_names = list(cancer.columns)
    variable_names.remove('ID')
    variable_names.remove('Diagnosis')
    resultIG = dict();
    for name in variable_names:
        resultIG[name] = split_gini(cancer[name], cancer['Diagnosis'])
        
    #Problem 3
    X = iris.data
    label, centers = kmeans(X, 3)
    scatter = np.hstack((X,np.array(label).reshape(-1,1)))
    plt.scatter(scatter[:,0], scatter[:,1], c=scatter[:,-1])
    
    #Problem 4
    rule = [{'a'},{'b'}]
    print(cal_support(trans,rule))
    print(cal_conf(trans, rule))
    rule = [{'b','c','e'},{'f'}]
    print(cal_support(trans,rule))
    print(cal_conf(trans, rule))
    rule = [{'a','c'},{'b','f'}]
    print(cal_support(trans,rule))
    print(cal_conf(trans, rule))
    rule = [{'b','d'},{'g'}]
    print(cal_support(trans,rule))
    print(cal_conf(trans, rule))
    rule = [{'b','e'},{'c','f'}]
    print(cal_support(trans,rule))
    print(cal_conf(trans, rule))
        
    #Problem 5
    X = iris.data
    y = iris.target
    components = PCA(X, 2)
    Xnew = X - X.mean(0)
    X_pca = np.matmul(Xnew, components)
    plt.scatter(X_pca[:,0], X_pca[:,1], c=y)

    
    # Apriori algorithm
    min_sup=0.4    
    Ck=generate_ck(trans,1)
    r=dict()
    for k in range(1,len(Ck)):
        Lk=generate_lk(trans,Ck,min_sup)
        r[k]=[Ck,Lk]
        Ck=generate_ck(trans,k+1,Lk)
        if len(Ck)==0:
            break
        
       
        
        
        
        
        
        
        
        
        
        
        
        