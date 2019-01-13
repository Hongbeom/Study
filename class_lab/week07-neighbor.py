# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:56:09 2018

@author: HongbeomChoi
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data=load_iris()
X=data.data
y=data.target

trainX, valX,trainY,valY=train_test_split(X,y,test_size=0.2,shuffle=True)

clf = KNeighborsRegressor(n_neighbors=3, metric='euclidean')#맨하탄은 cityblock이용

clf.fit(trainX, trainY)

y_pred = clf.predict(valX)

clf.score(valX,valY)

nb = clf.kneighbors(valX)


#유클리디안 디스턴스
def euclidean_dist(x,y):
    x=np.array(x)
    y=np.array(y)
    
    return np.sqrt(np.sum((x-y)**2))

#예시
euclidean_dist(valX[0],trainX[57])

#위의 함수를 써서 하는 예제
clf = KNeighborsClassifier(n_neighbors=10,metric=euclidean_dist)

clf.fit(trainX, trainY)

y_pred=clf.predict(valX)


#리그레션을 위한 데이터셋 만듬
xx=np.random.uniform(0,2*np.pi,100)
yy=np.sin(xx)+np.random.normal(scale=0.1,size=100)

import matplotlib.pyplot as plt

plt.scatter(xx,yy)

reg = KNeighborsRegressor(n_neighbors=50)
reg.fit(np.reshape(xx,(-1,1)),yy)

testX = np.linspace(-0.1,np.pi*2+0.1,100)
y_pred = reg.predict(np.reshape(testX,(-1,1)))

plt.scatter(xx,yy)
plt.plot(testX,y_pred,'r')

xx=np.random.uniform(0,1,size=(10,2))



from sklearn.neighbors import NearestNeighbors

nn=NearestNeighbors(n_neighbors=3)

p=[[0.5,0.5]]
nn.fit(xx)

nb=nn.kneighbors(p)
nb

plt.scatter(xx[:,0],xx[:,1])
plt.scatter(xx[nb[1],0],xx[nb[1],1],c='r',s=30)
plt.scatter([0.5],[0.5], marker='x',c='k',s=50)


xx[:,0]=10*xx[:,0]

plt.scatter(xx[:,0],xx[:,1])

nn.fit(xx)

nb = nn.kneighbors([[5,0.5]])

nb 

plt.scatter(xx[:,0],xx[:,1])
plt.scatter(xx[nb[1],0],xx[nb[1],1],c='r',s=30)
plt.scatter([5],[0.5], marker='x',c='k',s=50)

#standardization


xx2=(xx-xx.mean(0))/xx.std(0)

xx2.mean(0)
xx2.std(0)

from sklearn.preprocessing import scale

xx3=scale(xx)

from sklearn.neighbors import RadiusNeighborsClassifier, RadiusNeighborsRegressor

clf2=RadiusNeighborsClassifier()
clf2.fit(trainX,trainY)

nb=clf2.radius_neighbors(valX)

len(nb[1][0])
len(nb[1][1])

