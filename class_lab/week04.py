# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd 
import numpy as np
import os
datapath = r'C:\Users\HDC_USER\Desktop'

df = pd.read_csv(os.path.join(datapath, 'petrol_consumption.txt'), sep='\t', names=['tax', 'income','highway','driver','petrol'])


from sklearn.linear_model import LinearRegression



reg = LinearRegression()

X = df[['tax', 'income','highway','driver']]
y = df['petrol']

reg.fit(X, y)


error = y-reg.predict(X)


plt.hist(error, bins=10)
plt.hist(error, bins=20)
stats.probplot(error, dist='norm',plot=plt)

stats.skew(error)
stats.kurtosis(error)
stats.kurtosis(error, fisher=True)
stats.kurtosis(error, fisher=False)

stats.chi2.pdf
stats.chi2.cdf() 

e2=error**2
reg.fit(X,e2)

reg.score(X,e2)

reg.coef_
reg.intercept_

y_pred = reg.predict(X)

import matplotlib.pyplot as plt


x=np.arange(300,1001,100)
np.linspace(300, 1000, 8)
print(x)
l=x



plt.scatter(y, y_pred, s=50, c="gray")
plt.plot(x, l)

plt.scatter(df['income'], df['petrol'])
plt.xlabel('income')
plt.ylabel('petrol consumption')
plt.title('scatter plot')

y_pred.mean()
y_pred.std()
y_pred.var()


SSE=sum((y-y_pred)**2)
SSR=sum((y_pred-y_pred.mean())**2)

X.shape
n,p = X.shape


MSE=SSE/(n-p-1)
MSR=SSR/p

f=MSR/MSE


from scipy import stats

stats.f.pdf(f, p, n-p-1)
1-stats.f.cdf(f, p, n-p-1)


Xmat = np.c_[np.ones(n),X]

Xmat.T

XtX=np.matmul(Xmat.T, Xmat)
Xinv=np.linalg.inv(XtX)

t=reg.coef_[0]/np.sqrt(np.diag(Xinv)*MSE)[1]

(1-stats.t.cdf(np.abs(t), n-p-1))*2

reg.fit(df[['income','highway','driver']],df['tax'])

reg.score(df[['income','highway','driver']],df['tax'])

1/(1-reg.score(df[['income','highway','driver']],df['tax']))

blood = pd.read_csv(os.path.join(datapath, 'bloodpress.txt'), sep='\t',index_col=0)

cov=blood.cov()
cor=blood.corr()

X1=blood[['Age','Weight','BSA','Dur','Pulse','Stress']]
X2=blood[['Age','BSA','Dur','Pulse','Stress']]

y =blood['BP']

reg1= LinearRegression()
reg2 = LinearRegression()

reg1.fit(X1, y)
reg2.fit(X2,y)

reg1.coef_
reg2.coef_

reg.fit(blood[['Age','BSA','Dur','Pulse','Stress']], blood['Weight'])
1/(1-reg.score(blood[['Age','BSA','Dur','Pulse','Stress']], blood['Weight']))











