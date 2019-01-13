# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 18:18:37 2018

@author: Administrator
"""
# Use the following packages only
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import f as fdist
from scipy.stats import t as tdist
from scipy.stats import chi2 
from scipy.stats import stats
import matplotlib.pyplot as plt

def do_linear_regression(X,y):
    reg = LinearRegression()
    reg.fit(X,y)  
    return [reg.intercept_]+list(reg.coef_)

# Question 1
def predict(X, beta):
    ######## CALCULATE ESTIMATED TARGET WITH RESPECT TO X ########
    # INPUT
    # X: n by k (n=# of observations, k=# of input variables)
    # beta: vector of estimated coefficient by do_linear_regression
    #       beta[0] is intercept
    #       the remained elements correspond to variables in X
    # OUPUT
    # y_pred: 1D list(array) with length n, the estimated target
    
    # TODO: prediction
    y_pred = []
    y_pred = np.matmul(X, beta[1:]) + beta[0]
    return y_pred

# Question 2
def cal_SS(X, y, beta):
    ######## CALCULATE SST, SSR, SSE ########
    # INPUT
    # model: trained linear regression model
    # X: n by k (n=# of observations, k=# of input variables)
    # y: output target
    # beta: vector of estimated coefficient by do_linear_regression
    #       beta[0] is intercept
    #       the remained elements correspond to variables in X
    # OUTPUT
    # SST, SSR, SSE of the trained model
    
    # TODO: SS
    SST,SSR,SSE=0,0,0
    y_pred = predict(X, beta)
    SSR = sum((y_pred-y.mean())**2)
    SSE = sum((y_pred-y)**2)
    SST = SSR + SSE
    
    
    return (SST, SSR, SSE)


# Question 3
def f_test(X, y, beta, alpha):
    ######## PERFORM F-TEST ########
    # INPUT
    # X: n by k (n=# of observations, k=# of input variables)
    # y: output target
    # beta: vector of estimated coefficient by do_linear_regression
    #       beta[0] is intercept
    #       the remained elements correspond to variables in X
    # alpha: significant level
    # OUTPUT
    # f: f-test statistic of the model
    # pvalue: p-value of f-test
    # decision: f-test result 
    #           True = reject null hypothesis
    #           False = accept null hypothesis
    
    # TODO: F-test
    f = 0
    pvalue = 0
    decision = None
    n, p = X.shape
    SST, SSR, SSE = cal_SS(X, y, beta)
    MSR = SSR/p
    MSE = SSE/(n-p-1)
    
    f = MSR/MSE
    pvalue = 1-fdist.cdf(f, p, n-p-1)
    if(alpha <= pvalue):
        decision = False
    else:
        decision = True
        
    return (f,pvalue,decision)

# Question 4
def cal_tvalue(X,y,beta):
    ######## CALCULATE T-TEST TEST STATISTICS OF ALL VARIABES ########
    # INPUT
    # model: trained linear regression model
    # X: n by k (n=# of observations, k=# of input variables)
    # y: output target
    # beta: vector of estimated coefficient by do_linear_regression
    #       beta[0] is intercept
    #       the remained elements correspond to variables in X
    # OUTPUT
    # t: array(list) of size (k+1) with the t-test test statisc of variables (the first value is for intercept)
    
    # TODO: t-test statistics
    t = []
    n,p = X.shape
    SST, SSR, SSE = cal_SS(X,y,beta)
    X=np.c_[np.ones(n),X]
    XtX = np.matmul(X.T, X)
    XtXinv = np.linalg.inv(XtX)
    

    MSE = SSE/(n-p-1)
    
    se = np.sqrt(MSE*np.diag(XtXinv))
  
    t = np.array(beta).flatten()/se
    
    return t

# Question 5
def cal_pvalue(t,X):
    ######## CALCULATE P-VALUE OF T-TEST TEST STATISTICS ########
    # INPUT
    # t: array(list) of size (k+1) with the t-test test statisc of variables (the first value is for intercept)
    # X: n by k (n=# of observations, k=# of input variables)
    # OUTPUT
    # pvalue: array(list) of size (k+1) with p-values of t-test (the first value is for intercept)
    
    # TODO: p-value of t-test
    pvalue=[]
    n,p = X.shape
    for i in t:
        pvalue.append(1-tdist.cdf(abs(i), n-p-1))
    
    return pvalue

# Question 6
def t_test(pvalue,alpha):
    ######## DECISION OF T-TEST ########
    # INPUT
    # pvalue: array(list) of size (k+1) with p-values of t-test (the first value is for intercept)
    # alpha: significance level
    # OUTPUT
    # decision: array(list) of size (k+1) with t-test results of all variables
    #           True = reject null hypothesis
    #           False = accept null hypothesis
    # TODO: t-test 
    decision = []
    for i in pvalue:
        if(alpha/2 <= i):
            decision.append(False)
        else:
            decision.append(True)
    return decision

# Question 7
def cal_adj_rsquare(X,y,beta):
    ######## CACLULATE ADJUSTED R-SQUARE ########
    # INPUT
    # X: n by k (n=# of observations, k=# of input variables)
    # y: output target
    # beta: vector of estimated coefficient by do_linear_regression
    #       beta[0] is intercept
    #       the remained elements correspond to variables in X
    # OUTPUT
    # adj_rsquare: adjusted r-square of the model
    
    # TODO: adjusted r-square
    adj_rsquare=0 
    SST, SSR, SSE = cal_SS(X,y,beta)
    n,p = X.shape
    adj_rsquare = 1-(SSE/n-p-1)/(SST/n-1)
    return adj_rsquare

# Question 8
def skew(x):
    ######## CACLULATE Skewness ########
    # INPUT
    # x: 1D list (array)
    # OUTPUT
    # skew: skewness of the array x
    
    #TODO: calculate skewness
    #ONLY USE numpy
    skew = 0
    down = (np.mean(np.square(x-x.mean())))**(1.5)
    up = (np.mean((x-x.mean())**3))    
    
    skew = up/down
    
    return skew

# Question 9
def kurtosis(x):
    ######## CACLULATE Skewness ########
    # INPUT
    # x: 1D list (array)
    # OUTPUT
    # kurt: kurtosis of the array x
    
    #TODO: calculate kurtosis
    #ONLY USE numpy
    kurt = 0
    
    down = (np.mean(np.square(x-x.mean())))**2
    up = np.mean((x-x.mean())**4)
    kurt = up/down
    
    return kurt

# Question 10
def jarque_bera(X,y,beta,alpha):
    ######## JARQUE-BERA TEST ########
    # INPUT
    # model: trained linear regression model
    # X: n by k (n=# of observations, k=# of input variables)
    # y: output target
    # beta: vector of estimated coefficient by do_linear_regression
    #       beta[0] is intercept
    #       the remained elements correspond to variables in X
    # alpha: significance level
    # OUTPUT
    # JB: Jarque-Bera test statistic
    # pvalue: p-value of the test statistic
    # decision: Jarque-Bera test result 
    #           True = reject null hypothesis
    #           False = accept null hypothesis
    
    # TODO: Jarque-Bera test
    JB = 0
    pvalue =0
    decision = None
    n = len(y)
    y_pred = predict(X, beta)
    error = y-y_pred
    s = skew(error)
    c = kurtosis(error)
    k,p = X.shape
    JB = ((n-p)/6)*((s**2)+((c-3)**2)/4)
    
    pvalue = 1-chi2.cdf(JB, 2)
    if(alpha <= pvalue):
        decision = False
    else:
        decision = True
    return (JB,pvalue,decision)

# Question 11
def breusch_pagan(X,y,beta,alpha):
    ######## BREUSCH-PAGAN TEST ########
    # INPUT
    # X: n by k (n=# of observations, k=# of input variables)
    # y: output target
    # beta: vector of estimated coefficient by do_linear_regression
    #       beta[0] is intercept
    #       the remained elements correspond to variables in X
    # alpha: significance level
    # OUTPUT
    # LM: Breusch-pagan Lagrange multiplier statistic
    # pvalue: p-value of the test statistics
    # decision: Breusch-pagan test result 
    #           True = reject null hypothesis
    #           False = accept null hypothesis
    
    # TODO: Breusch-pagan test
    LM = 0
    pvalue = 0
    decision = None
    n,p = X.shape
    y_pred = predict(X, beta)
    error = y-y_pred
    e2 = error**2
    beta2 = do_linear_regression(X,e2)
    
    SST, SSR, SSE = cal_SS(X, e2, beta2)
    print(SST,SSR,SSE)
    R2 = SSR/SST
    LM = n*R2
    pvalue = 1-chi2.cdf(LM, p-1)
    if(alpha <= pvalue):
        decision = False
    else:
        decision = True

    return (LM,pvalue,decision)


if __name__=='main':
    # LOAD DATA
    data = pd.read_csv('https://drive.google.com/uc?export=download&id=1YPnojmYq_2B_lrAa78r_lRy-dX_ijpCM', sep='\t')
    # INPUT
    X = data[data.columns[:-1]]
    # TARGET
    y = data[data.columns[-1]]
    
    alpha = 0.05
    coefs= do_linear_regression(X,y)
    coefs
    #################### TEST YOUR CODE ####################
    breusch_pagan(X,y,coefs,alpha)
    
