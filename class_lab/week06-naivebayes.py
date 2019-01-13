# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 13:30:22 2018

@author: user
"""

from sklearn.datasets import load_iris

data=load_iris()

X=data.data
y=data.target

C1=X[y==0]
# 여기 혜인이가 질문해서 잠깐 보여준거

#여기가 랩 시작인듯
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

import numpy as np

np.mean(X)
np.mean(X, axis=1)
np.mean(X, axis=0)

np.mean(X[:,1])
np.std()
np.median()


Xbin=(X>np.mean(X, axis=0))*1

berNB = BernoulliNB(alpha=2)
berNB.fit(Xbin, y)

berNB.coef_
np.exp(berNB.coef_)

xll=Xbin[y==0,0]


GNB=GaussianNB()
GNB.fit(X,y)

GNB.sigma_
GNB.theta_

y_pred_G=GNB.predict(X)
y_pred_ber=berNB.predict(Xbin)

GNB.score(X,y)
berNB.score(Xbin,y)

import matplotlib.pyplot as plt

for i in np.unique(y):
    plt.hist(X[y==i,0], bins=10, label=str(i), alpha=0.5)
plt.legend()

xx=np.linspace(4,8,100)

from scipy import stats



mu=GNB.theta_[0,0]
sigma=GNB.sigma_[0,0]
yy=stats.norm.pdf(xx,loc=mu, scale=sigma)

plt.hist(X[y==0,0], bins=10, label=str(i), alpha=0.5,density=True)
plt.plot(xx, yy)


multiNB=MultinomialNB()

Xmulti=np.round(X)

multiNB.fit(Xmulti,y)
multiNB.coef_
multiNB.score(Xmulti,y)

p=np.exp(multiNB.coef_)
np.sum(p,axis=1)
y_prob=GNB.predict_proba(X)


import pandas as pd

sms=pd.read_csv(r'C:\Users\user\Downloads\spam_sms.csv')


multiNB.fit(sms[sms.columns[:-1]],sms['target'])
multiNB.score(sms[sms.columns[:-1]],sms['target'])

p=np.exp(multiNB.feature_log_prob_)
p.shape

np.unique(sms['target'])
np.sum(p,axis=1)
