# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 02:59:16 2020

@author: Ratnesh
"""

import cv2
import os
import numpy as np
from sklearn.metrics import accuracy_score

#data loading
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#            cv2.imshow('Gray image', img)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
            images.append(gray.flatten())
    return images

path = os.getcwd() + "\\training_faces"
training_images = np.zeros((6,10304))
for n,folder in enumerate(os.listdir(path)):
    if(n==0):
        training_images+=(load_images_from_folder(path+"\\"+folder))
    else:
        training_images = np.row_stack((training_images,(load_images_from_folder(path+"\\"+folder))))

path = os.getcwd() + "\\testing_faces"
testing_images = np.zeros((4,10304))
for n,folder in enumerate(os.listdir(path)):
    if(n==0):
        testing_images+=(load_images_from_folder(path+"\\"+folder))
    else:
        testing_images = np.row_stack((testing_images,(load_images_from_folder(path+"\\"+folder))))

#---------------------------------------------Training-Begins-----------------------------------------------------#
#data pre-processing
#zero-out-mean
mean = np.mean(training_images,axis=0)
training_images = training_images - mean

#Finding co-variance matrix
cov = (training_images @ training_images.T)/training_images.shape[1]

eigen_val,eigen_vec = np.linalg.eig(cov)

mydic = dict(zip(eigen_val,eigen_vec))
k = 15

feature_vec = np.zeros((k,training_images.shape[0]))
for n,i in enumerate(sorted(mydic,reverse=True)):
    if (n <k):
        feature_vec[n] += mydic[i]

eigen_faces = feature_vec @ training_images

sign_faces = eigen_faces @ training_images.T
img_class = {}
for a in range(10):
    img_class[a]=((sign_faces.T)[a*6:(a+1)*6,:])

mean_class = np.array([np.mean(img_class[i],axis = 0) for i in sorted(img_class)])
mean_proj =  np.mean(sign_faces.T,axis = 0)

cov_within_class = {}
for a in sorted(img_class):
    temp1 = img_class[a] - mean_class[a]
    cov_within_class[a] = temp1.T @ temp1

SW = np.zeros(cov_within_class[0].shape)

#calculating Scattered Within Matrix
for a in sorted(cov_within_class):
    SW += cov_within_class[a]

SB = np.zeros(SW.shape)
for a in sorted(img_class):
    temp2 = (mean_class[a] - mean_proj)
    SB += (temp2.T @ temp2)

#criterion function calculation

J = (np.linalg.inv(SW)) @ SB

eig_val,eig_vec = np.linalg.eig(J)

m = 7
#mydic1 = dict(zip(eig_val,(eig_vec)))

#best_featc = np.zeros((m,k),complex)
#for n,i in enumerate(sorted(mydic1,reverse=True)):
#    if (n < m):
#        best_featc += mydic1[i]
order = np.flip(np.argsort(eig_val))
eig_vec = eig_vec[order]
best_featc = np.real(eig_vec[:m])

#fisher faces calculation
fisher_faces = best_featc @ sign_faces

#----------------------------------------------Training-Ends-------------------------------------------------------#
def mahalanobis(x, data,S):
    diff = (x - data).reshape((x.shape[0], 1))
    dist = np.sqrt(diff.T @ S @ diff)
    return dist[0][0]


#---------------------------------------------Testing-Begins-------------------------------------------------------#

cov_inv =np.linalg.inv(np.cov(fisher_faces))
temp = np.arange(10)

#number of time a same subject has image: 4
temp = np.repeat(temp,4)
testing_images = np.column_stack((temp,testing_images))
np.random.shuffle(testing_images)

Y = (testing_images.T)[0]
testing_images = testing_images[:,1:]

pred_y = np.zeros(testing_images.shape[0])
pred_y1 = np.zeros(testing_images.shape[0])

#projecting testing data into eigen faces
for n,img in enumerate(testing_images):
    proj = eigen_faces @ (img - mean).T
    fisher_proj = best_featc @ proj
    dist = (fisher_proj - fisher_faces.T)
    dist = np.linalg.norm(dist.T,axis = 0)
    index = np.argmin(dist)
    index1 = np.argmin(np.array([mahalanobis(fisher_proj, img, cov_inv) for img in fisher_faces.T]))
    pred_y[n] = np.floor(index/6)
    pred_y1[n] = np.floor(index1/6)

print("Using Euclidean:", accuracy_score(Y,pred_y))
print("Using Mahalanobis:",accuracy_score(Y,pred_y1))

#----------------------------------------------Testing-Ends-------------------------------------------------------#