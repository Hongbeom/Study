# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\idd74\Desktop\icecream_sale.txt', sep='\t', names=['temp','sales'], dtype = {'sales':'float'})


df.head(10)
df.tail()

df.dtypes

df=pd.Series([1,2,3,4,5])
df

df=pd.DataFrame([[1,2],[3,4]], columns=['X1','X2'], index=['A','B'])

df[['X1','X2']]

df.loc['A','X2']
df.loc['A',['X1', 'X2']]

df.iloc[0, 0]
df.iloc[-1, -1]

df.index
df.columns
df.values

df['X1'].mean()
df['X1'].std()
df['X1'].var()
df['X1'].quantile(0.5)

df['X1'].describe()

df.sort_values('temp', ascending=False)


df['temp']>20

df[df['temp']>20]

df.iloc[0,1] = 200

df=pd.DataFrame([[1, np.nan],[3,4]], columns=['X1','X2'], index=['A','B'])

df.dropna()
df.fillna(10)





from sklearn.linear_model import LinearRegression

df = pd.read_csv(r'C:\Users\idd74\Desktop\icecream_sale.txt', sep='\t', names=['temp','sales'], dtype = {'sales':'float'})

reg = LinearRegression()

X = df[['temp']]
y = df['sales']
Z = df['temp']

df['temp'].toframe()
df['temp'].reshape((-1,1))

reg.fit(X,y)

reg.coef_
reg.intercept_

X*reg.coef_+reg.intercept_

reg.predict(X)

petrol =pd.read_csv(r'C:\Users\idd74\Desktop\petrol_consumption.txt', sep='\t', names=['tax','income', 'highway', 'driver', 'petrol'])

X = petrol[['tax','income', 'highway', 'driver']]
y = petrol['petrol']

reg.fit(X,y)

reg.coef_
reg.intercept_

np.zeros((3,3))
np.ones((2,1))

n,p=X.shape

X=np.c_[np.ones(n),X]

XtX = np.matmul(X.T, X)

XtXinv = np.linalg.inv(XtX)

beta = np.matmul(np.matmul(XtXinv, X.T), np.reshape(y.to_frame(), (-1,1)) )

beta[0]
reg.intercept_

y_pred = np.matmul(X, beta)

y_pred = y_pred.flatten()

SSE = sum((y-y_pred)**2)
SSR = sum((y_pred-y.mean())**2)

MSR=SSR/p
MSE=SSE/(n-p-1)

f = MSR/MSE
f

from scipy import stats

stats.f.cdf(f, p, n-p-1)

np.diag(XtXinv)

se = np.sqrt(MSE*np.diag(XtXinv))

t = beta.flatten()/se

stats.t.cdf(abs(t[3]), n-p-1)


SSR/(SSR+SSE)

reg.score(X,y)
