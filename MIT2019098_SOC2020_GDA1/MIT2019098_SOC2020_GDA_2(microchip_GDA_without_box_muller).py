# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 23:58:26 2020

@author: Ratnesh
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 20:13:01 2020

@author: Ratnesh
"""

import pandas as pd
import numpy as np
from numpy import sqrt,pi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,accuracy_score
sc = MinMaxScaler()
dataset = pd.read_csv("microchip_dataset.csv")

X = dataset.iloc[:,:2].values
train, test = train_test_split(dataset, test_size=0.5, shuffle = True,random_state = 1)

train0 = train[train["result"] == 0]
train1 = train[train["result"] == 1]
train_Y = (train.iloc[:,-1].values).reshape(train.shape[0],1)
train_X = train.iloc[:,:2].values
test_X = test.iloc[:,:2].values
test_Y = (test.iloc[:,-1].values).reshape(test.shape[0],1)
#find the mean for y = 0 and y = 1
X0 = train0.iloc[:,:2].values
X1 = train1.iloc[:,:2].values

mean0 = (np.mean(X0,axis = 0)).reshape(1,X0.shape[1])
mean1 = (np.mean(X1,axis = 0)).reshape(1,X1.shape[1])

#finding co-variance_matrix
#diff x and mean, for convenience we are taking mean0
diff = train_X - mean0
cov = (1/train_X.shape[0])*(diff.T @ diff)

#calculate p(x|y)
def calc_px_y0_1(x,mean0,mean1,cov):
    n = x.shape[0]
    c = 1/(((2*pi)**(n/2))*(sqrt(np.linalg.det(cov))))
    diff0 = (x - mean0)
    diff1 = (x - mean1)
    p_x_y_0 = (c*np.exp(-0.5*(diff0@np.linalg.inv(cov))@(diff0.T)))
    p_x_y_1 = (c*np.exp(-0.5*(diff1@np.linalg.inv(cov))@(diff1.T)))
    return p_x_y_0,p_x_y_1

def calc_phi(m,x1):
    n = x1.shape[0]
    return n/m;

def calc_py(y,phi):
    if(y == 1):
        return phi
    else:
        return (1 - phi)

#calculate p(y=1/0 |x)
def calc_pred_y(x,X1,mean1,mean0,cov):
    p_x_y_0 = np.zeros(x.shape[0])
    p_x_y_1 = np.zeros(x.shape[0])

    for a in range(x.shape[0]):
        p_x_y_0[a],p_x_y_1[a] = calc_px_y0_1(x[a],mean0,mean1,cov)
        #p_x_y_0[a],p_x_y_1 = calc_px_y0_1(test_X,mean0,mean1,cov)
    phi = calc_phi(train_X.shape[0],X1)
    py0 = calc_py(0,phi)
    py1 = calc_py(1,phi)
    p_y_0_x = p_x_y_0*py0
    p_y_1_x = p_x_y_1*py1
    pred_y = [1 if p_y_1_x[a] > p_y_0_x[a] else 0 for a in range(p_y_1_x.shape[0])]
    return pred_y

pred_y = calc_pred_y(test_X,X1,mean1,mean0,cov)
#print(accuracy_score(train_Y,pred_y))
#print(confusion_matrix(train_Y,pred_y))
print(accuracy_score(test_Y,pred_y))
print(confusion_matrix(test_Y,pred_y))
