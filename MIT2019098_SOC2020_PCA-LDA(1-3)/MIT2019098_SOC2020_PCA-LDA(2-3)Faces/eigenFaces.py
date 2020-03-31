# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 22:50:16 2020

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
k = 10

feature_vec = np.zeros((k,60))
for n,i in enumerate(sorted(mydic,reverse=True)):
    if (n <k):
        feature_vec[n] += mydic[i]

eigen_faces = feature_vec @ training_images

sign_faces = eigen_faces @ training_images.T

#----------------------------------------------Training-Ends-------------------------------------------------------#

#---------------------------------------------Testing-Begins-------------------------------------------------------#

testing_images = testing_images - mean
temp = np.arange(10)
#number of time a same subject has image: 4
temp = np.repeat(temp,4)
testing_images = np.column_stack((temp,testing_images))
np.random.shuffle(testing_images)

Y = (testing_images.T)[0]

testing_images = testing_images[:,1:]

pred_y = np.zeros(testing_images.shape[0])

#projecting testing data into eigen faces
false_found = 0
false_match = 0
total_false = 0
print("Testing\n")
for n,img in enumerate(testing_images):
    proj = eigen_faces @ img.T
    dist = (proj - sign_faces.T)
    dist = np.linalg.norm(dist.T,axis = 0)
    max_index = np.argmax(dist)
    min_index = np.argmin(dist)
    min_dist = dist[min_index]
    max_dist = dist[max_index]
#    print(min_dist/max_dist)
    #setting threshold for identifying false acceptance
    if((min_dist/max_dist)*100 > 27):
        print("not enrolled")
        false_found +=1
        #pseudo correct prediction as not accpeted
        pred_y[n] = np.floor(min_index/6)
    else:
        pred_y[n] = np.floor(min_index/6)
        if(pred_y[n] != Y[n]):
            false_match += 1
print("Test Accuracy:", accuracy_score(Y,pred_y))
#print("False Accepted Rate(FAR):",abs(total_false - false_found)/Y.shape[0])
#print("False Matched Rate(FMR):",false_match/Y.shape[0])
#---------------------------------------------Testing-For Imposter-------------------------------------------------------#

#for imposter who does not belong to any group
path = os.getcwd() + "\\Imposter"
imposter_images = np.zeros((10,10304))
imposter_images+=(load_images_from_folder(path))

imposter_images = imposter_images - mean
temp = np.arange(11,21,1).reshape((imposter_images.T).shape[1],1)
imposter_images = np.column_stack((temp,imposter_images))
np.random.shuffle(imposter_images)

Y = (imposter_images.T)[0]
imposter_images = imposter_images[:,1:]

pred_y1 = np.zeros(imposter_images.shape[0])
print("\nImposter Testing:\n")
#projecting testing data into eigen faces
false_found = 0
false_match = 0
total_false = 10
for n,img in enumerate(imposter_images):
    proj = eigen_faces @ img.T
    dist = (proj - sign_faces.T)
    dist = np.linalg.norm(dist.T,axis = 0)
    max_index = np.argmax(dist)
    min_index = np.argmin(dist)
    min_dist = dist[min_index]
    max_dist = dist[max_index]
#    print(min_dist/max_dist)
    if((min_dist/max_dist)*100 > 15):
        print("subject",n,"not enrolled\n")
        false_found +=1
        pred_y1[n] = Y[n]
    else:
        pred_y1[n] = np.floor(min_index/6)
        if(pred_y[n] != Y[n]):
            false_match += 1

print("False Accepted Rate(FAR):",(total_false - false_found)/Y.shape[0])