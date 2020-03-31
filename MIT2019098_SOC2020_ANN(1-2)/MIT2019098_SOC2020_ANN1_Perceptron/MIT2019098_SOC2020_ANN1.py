# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 14:13:28 2020

@author: Ratnesh
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score

def create_train_data(gate):
    train_X1 = np.array([0,0,1,1])
    train_X2 = np.array([0,1,0,1])
    temp = np.array([1,1,1,1])
    X = np.column_stack((temp,train_X1,train_X2))
    if(gate=="AND"):
        return X,np.array([0,0,0,1])
    elif(gate=="OR"):
        return X,np.array([0,1,1,1])
    elif(gate=="NOR"):
        return X,np.array([1,0,0,0])
    elif(gate == "NAND"):
        return X,np.array([1,1,1,0])
    elif(gate == "EX-OR"):
        return X,np.array([0,1,1,0])
    else:
        return X,np.array([1,0,0,1])

def NOT(x):
    z = np.zeros(x.shape)
    for i,a in enumerate(x):
        if(a):
            z[i] = 0
        else:
            z[i] = 1
    return z

def step(x):
    if(x>=0):
        return 1
    else:
        return 0
def train_percep(gate,epoch,alpha):
    error_arr = [0]
    X,Y = create_train_data(gate)
    W_pres = np.array([0.2,0.2,0.2])
    W_next = np.copy(W_pres)
    for i in range(epoch):
        errors = 0
        for xi, y_des in zip(X,Y):
            z = step(xi @ W_pres)
            delta = (y_des - z)
            if(delta!=0):
                errors+=1
                W_pres += alpha*xi*delta
        error_arr.append(errors)
        if(np.fabs(np.sum(W_pres - W_next)) <= 0.001):
            break
        W_next = np.copy(W_pres)
    plt.title("Error_"+gate)
    plt.plot(error_arr)
    plt.xlabel('number of epoch')
    plt.ylabel('number of error')
    plt.show()
    return W_pres

def test_perc(W,x):
    pred_Y = []
    for xi in x:
        z = xi @ W
        pred_Y.append(step(z))
    return pred_Y

W_AND = train_percep("AND",50,0.1)
W_OR =  train_percep("OR",50,0.1)
W_NOR =  train_percep("NOR",50,0.1)
W_NAND =  train_percep("NAND",50,0.1)

#create test data using AND and OR pred result and verifying against EX-OR
#EX_OR =  (A AND NOT B) OR (B AND NOT A)
test_X,test_Y = create_train_data("EX-OR")
test_X1 = np.copy(test_X.T)
X_1 = NOT((test_X1)[2])
test_X1[2] = X_1

test_X2 = np.copy(test_X.T)
X_2  = NOT((test_X2)[1])
test_X2[1] = test_X2[2]
test_X2[2] = X_2
#pred_a_and_not_b: Say p
#pred_b_and_not_a: Say q
p = test_perc(W_AND,test_X1.T)
q = test_perc(W_AND,test_X2.T)
pq = np.column_stack(((np.ones(test_X.shape[0])),p,q))
pred_y = test_perc(W_OR,pq)
print(accuracy_score(test_Y,pred_y))