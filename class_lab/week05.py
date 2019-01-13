# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 13:38:52 2018

@author: user
"""

def HappyBirty(person):
    print("Happy Birthday,%s"%(person))

HappyBirty("Hongbeom")

def math(a,b,c=1):
    return c*(a+b)

math(15,20, c=3)

import numpy as np

def bernoulli(D,p):
    L=1
    for d in D:
        L*=(p**(d))*(1-p)**(1-d)
    return L

def bernoulli(D,p):
    D=np.array(D)
    return np.prod((p**D)*((1-p)**(1-D)))

D=[1,0,1,1,1,0]        

1-np.array(D)

bernoulli(D, 0.3)
bernoulli(D, 0.6)

np.prod([1,2,3])

np.pi
np.exp(-1)
np.exp(1)

np.log(np.exp(1))
np.log2(2**4)

np.sqrt(3)


import pandas as pd

df=pd.read_csv(r"C:\Users\user\Downloads\height.txt",sep='\t',names=['X','Y'])
df.head()

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

clf=LogisticRegression()

df=df.sort_values(['Y'])
t=(df['Y']=='male')*1

n=len(df)
plt.scatter(range(n),df['X'], c=t)

clf.fit(df[['X']], df['Y'])

clf.coef_
clf.intercept_

clf.predict(df[['X']])

clf = LogisticRegression(C=10e5)
clf.fit(df[['X']],df['Y'])

clf.coef_
clf.intercept_

y_pred = clf.predict(df[['X']])

sum(y_pred==df['Y'])/n

clf.score(df[['X']], df['Y'])


y_prob=clf.predict_proba(df[['X']])
y_prob

from sklearn.datasets import load_iris

iris=load_iris()
X=iris['data']
y=iris['target']

clf=LogisticRegression()
clf.fit(X,y)

clf.coef_
clf.intercept_

clf=LogisticRegression(multi_class='multinomial',solver='lbfgs')
clf.fit(X,y) 

y_prob=clf.predict_proba(X)
y_prob
