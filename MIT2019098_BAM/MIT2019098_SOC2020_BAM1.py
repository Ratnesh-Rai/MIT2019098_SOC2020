# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 22:58:41 2020

@author: Ratnesh
"""

import numpy as np

data_pair = [[[1,1,1,1,1,1],[1,1,1]],
             [[-1,-1,-1,-1,-1,-1],[-1,-1,-1]],
             [[1,-1,-1,1,1,1],[-1,1,1]],
             [[1,1,-1,-1,-1,-1],[1,-1,1]]]

X = np.array((data_pair[0][0],data_pair[1][0],data_pair[2][0],data_pair[3][0]))
Y = np.array((data_pair[0][1],data_pair[1][1],data_pair[2][1],data_pair[3][1]))

len_x = len(data_pair[0][0])
len_y = len(data_pair[0][1])


#initial weight Matrix Calculation
def calc_bam_mat(data_pair):
    W = np.zeros((len_x,len_y))
    for pair in data_pair:
        W += (np.array(pair[0])).reshape(len_x,1) @ ((np.array(pair[1])).reshape(len_y,1)).T
    return W


#transmission Function
def threshold(A):
    ret_vec = []
    for i in A:
        if(i >= 0):
            ret_vec.append(1)
        else:
            ret_vec.append(-1)
    return ret_vec

def feed_backward(out,W):
   return threshold((out @ W.T))

def feed_forward(inp,W):
    result = (inp @ W)
    return threshold(result)

W = calc_bam_mat(data_pair)

def update_weight(W1,V1,X,Y,l_rate):
    y_pred = np.zeros(Y.shape)
    x_pred =  np.zeros(X.shape)
    for i in range(X.shape[0]):
        xi = X[i]
        yi = Y[i]
        y_pred[i] = (feed_forward(xi,W))
        x_pred[i] =(feed_backward(yi,W))

    W1 += l_rate*((Y - y_pred).T @ (X + x_pred)).T
    V1 += l_rate*((X - x_pred).T @ (Y + y_pred)).T
    return W1,V1

#training
epoch = 3
V = (W.copy()).T
for i in range(epoch):
    W1,V1 = update_weight(W.copy(),V.copy(),X,Y,0.1)
    print(i,W1)
    if((W == W1).all() and (V == V1).all()):
        break
    W = W1
    V = V1

#testing
print("\nY_pred: ",feed_forward(np.array((-1, -1, 1, 1, 1, 1)),W),"\nY-expected:",Y[2])
print("\nX_pred: ",feed_backward(np.array((-1,1,-1)),V.T),"\nX-expected: ",X[1])
