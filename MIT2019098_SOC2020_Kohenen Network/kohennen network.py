# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 12:56:21 2020

@author: Ratnesh
"""

import numpy as np

np.random.seed(17)
# number of input data
n = 1500
# number of nuerons(nodes) in kohennen network
n_nuerons = 100
# generating random uniform input between(-1, 1)
x1 = np.random.uniform(-1, 1, n)
x2 = np.random.uniform(-1, 1, n)

X = np.column_stack((x1, x2))

# initialising random weights to each nuerons
i = 0
W = np.array([np.random.uniform(-1, 1, 2) for i in range(n_nuerons)])

# this function trains the kohnnen network and returns the association list of
# node with input data and trained weight of each node


def train(l_rate, epoch, W, X):
    # Initialising change counter with some larger value
    # will be using this counter for checking convergence
    ch_cnt = 10000
    cnt = 1
    association = np.zeros(X.shape[0])  # will be used to store the association
    while(epoch):
        if(ch_cnt <= 1):
            break
        ch_cnt = 0
        epoch -= 1
        for i, x in enumerate(X):
            diff = []
            for w in W:
                diff.append(np.linalg.norm(x - w))
            index = np.argmin(diff)
            W[index] += l_rate*(x - W[index])
            if(index != association[i]):
                association[i] = index
                ch_cnt += 1
        print("epoch, change_count:", cnt, ch_cnt)
        cnt += 1
    print("\nTraining Completed in :", cnt - 1)
    return W, association


l_rate = 0.1
epoch = 100
print("\n======================Training=============================")
W, association = train(l_rate, epoch, W, X)

# print("\nX \t\t: Associated Node Index")
# for i in range(X.shape[0]):
#     print(X[i], ":\t", association[i])

testing_data = np.array(((0.1, 0.8), (0.5, 0.2), (-0.8, -0.9), (-0.06, 0.9)))
associated = []
for x in testing_data:
    diff = []
    for w in W:
        diff.append(np.linalg.norm(x - w))
    index = np.argmin(diff)
    associated.append(index)

print("\n\n======================Testing=============================")
print("X\t\t:\t Associated Node Index")

for i in range(testing_data.shape[0]):
    print(testing_data[i], "\t:", associated[i])
