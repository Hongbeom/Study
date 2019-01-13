# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 18:47:40 2018

@author: Administrator
"""
# Use the following packages only
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def logistic_reg(X,y):
    clf=LogisticRegression()
    clf.fit(X,y)
    return list(clf.intercept_)+list(clf.coef_[0])

# QUESTION 1
def cal_logistic_prob(X,y,beta):
    ######## CALCULATE PROBABILITY OF THE CLASS ########
    # INPUT
    # X: n by k (n=# of observations, k=# of input variables)
    # y: output target (Assumption: binary classification problem)
    # beta: array(list) of size (k+1) with the estimated coefficients of variables (the first value is for intercept)
    #        This coefficiens are for P(Y=k) where k is the larger number in output target variable
    # OUTPUT
    # p: probability of P(Y=k) where k is the larger number in output target variable
    
    # TODO: calculate proability of the class with respect to the given X for logistic regression
    n,p = X.shape
    matX = np.c_[np.ones(n), X]
    reg = np.matmul(matX, beta)
    p=[]
    for i in reg:
        p.append(1/(1+np.exp(-1*i)))
    

    
    return p

# QUESTION 2
def cal_logistic_pred(y_prob,cutoff,classes):
    ######## ESTIMATE OUTPUT CLASS ########
    # INPUT
    # y_prob: probability of P(Y=k) where k is the larger number in output target variable
    # cutoff: threshold for decision
    # classes: labels of classes
    # OUTPUT
    # y_pred: array(list) with the same size of y_prob, estimated output class 
    
    # TODO: estimate output class based on y_prob and cutoff (logistic regression)
    # if probability>cutoff → classes[1] else classes [0]
    y_pred=[]
    for i in y_prob:
        if (i>cutoff):
            y_pred.append(classes[1])
        else:
            y_pred.append(classes[0])
    return y_pred

# QUESTION 3    
def cal_acc(y_true,y_pred):
    ######## CALCULATE ACCURACY ########
    # INPUT
    # y_true: array(list), true class
    # y_pred: array(list), estimated class
    # OUPUT
    # acc: accuracy
    
    # TODO: calcuate accuracy
    acc = 0
    up = 0
    y_true = np.array(y_true)
    for i in range(len(y_true)):
        if(y_true[i]==y_pred[i]):
            up +=1
    acc = up/len(y_true)
    return acc

# QUESTION 4   
def BNB(X,y):
    ######## BERNOULLI NAIVE BAYES ########
    # INPUT 
    # X: n by p array (n=# of observations, p=# of input variables)
    # y: output (len(y)=n, categorical variable)
    # OUTPUT
    # pmatrix: 2-D array(list) of size c by p with the probability p_ij where c is number of unique classes in y
        
    # TODO: Bernoulli NB
    pmatrix=[]
    pc1 = []
    pc2 = []
    for c in X.columns:
        avg = X[c].mean()
        one_1 = 0
        zero_1 = 0
        one_2 = 0
        zero_2 = 0
        for i in X.index:
            if(y[i]==1):    
                if(X[c][i]>avg):
                    one_1 += 1
                else:
                    zero_1 += 1
            else:
                if(X[c][i]>avg):
                    one_2 += 1
                else:
                    zero_2 += 1
        pc1.append(one_1/(one_1+zero_1))
        pc2.append(one_2/(one_2+zero_2))
    
    pmatrix.append(pc1)
    pmatrix.append(pc2)

    
         
    return pmatrix

# QUESTION 5
def cal_BNB_prob(X,prior,pmatrix):
    ######## CALCULATE PROBABILITY OF THE CLASS ########
    # INPUT 
    # X: n by p array (n=# of observations, p=# of input variables)
    # priors: 1D array of size c where c is number of unique classes in y, prior probabilities for classes
    # pmatrix: 2-D array(list) of size c by p with the probability p_ij where c is number of unique classes in y
    # OUTPUT
    # p: n by c array, p_ij stores P(y=cj|X_i)
       
        
    # TODO: calculate proability of the class with respect to the given X for Bernoulli NB
    X2=X.copy()
    class_1 = []
    class_2 = []
    for c in X2.columns:
        avg = X2[c].mean()
        for i in X2.index:
                if(X2[c][i]>avg):
                    X2.ix[i][c] = 1
                else:
                    X2.ix[i][c] = 0
                    
    for i in X2.index:
        c1 = 1
        c2 = 1
        w = 0
        for c in X2.columns:
            c1 *= (pmatrix[0][w])**X2[c][i]
            c2 *= (pmatrix[1][w])**X2[c][i]
            w += 1
        class_1.append(c1*prior[0])
        class_2.append(c2*prior[1])
        
            
    p=[]
    p.append(class_1)
    p.append(class_2)
    p=np.array(p).T
    return p



# QUESTION 6
def cal_BNB_pred(y_prob,classes):
    ######## ESTIMATE OUTPUT CLASS ########
    # INPUT
    # y_prob: probability of P(Y=k) where k is the larger number in output target variable
    # classes: labels of classes
    # OUTPUT
    # y_pred: array(list) with the same size of y_prob, estimated output class 
    
    # TODO: estimate output class based on y_prob (Bernoulli NB)
    y_pred=[]
    
    for i in range(len(y_prob)):
        y = y_prob[i][:].tolist()
        y_pred.append(classes[y.index(max(y))])
    
    return y_pred
    
# QUESTION 7
def euclidean_dist(a,b):
    ######## EUCLIDEAN DISTANCE ########
    # INPUT
    # a: 1-D array 
    # b: 1-D array 
    # a and b have the same length
    # OUTPUT
    # d: Euclidean distance between a and b
    
    # TODO: Euclidean distance
    value = 0
    for x,y in zip(a,b):
        value += (x-y)**2
        
    d = 0
    
    d = np.sqrt(value)
    return d

# QUESTION 8
def manhattan_dist(a,b):
    ######## EUCLIDEAN DISTANCE ########
    # INPUT
    # a: 1-D array 
    # b: 1-D array 
    # a and b have the same length
    # OUTPUT
    # d: Manhattan distance between a and b
    
    # TODO: Manhattan distance
    d = 0
    for x,y in zip(a,b):
        d += abs(x-y)
        
    return d

# QUESTION 9
def knn(trainX,trainY,testX,k,dist=euclidean_dist):
    ######## K-NN Classification ########
    # INPUT 
    # trainX: training input dataset, n by p size 2-D array
    # trainY: training output target, 1-D array with length of n
    # testX: test input dataset, m by p size 2-D array
    # k: the number of the nearest neighbors
    # dist: distance measure function
    # OUTPUT
    # y_pred: predicted output target of testX, 1-D array with length of m
    #         When tie occurs, the final class is select in alpabetical order
    #         EX) if "A" ties "B", select "A" and if "2" ties "4", select 2
    
    # TODO: k-NN classification

    
    y_pred = []
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    testX = np.array(testX)
    
    for m in range(len(testX)):
        test = testX[m][:]
        distance = []
        index = []
        for n in range(len(trainX)):
            train = trainX[n][:]
            distance.append(dist(train,test))
        order = np.argsort(distance)[:k]
        for i in order:
            index.append(trainY[i])
        unique, counts = np.unique(index, return_counts=True)
        count=counts.tolist()
        y_pred.append(unique[count.index(max(count))])
    
    return y_pred

if __name__=='main':
    data=pd.read_csv(r'https://drive.google.com/uc?export=download&id=1QhUgecROvFY62iIaOZ97LsV7Tkji4sY4',names=['ID','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Y'])
    y=data['Y']
    X=data.loc[(y==1)|(y==2),['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']]
    y=y.loc[(y==1)|(y==2)]
    
    trainX,testX,trainY,testY=train_test_split(X,y,test_size=0.2,random_state=11)
   
    
    
