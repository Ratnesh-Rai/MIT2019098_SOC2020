# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 19:30:31 2020

@author: Ratnesh
"""
import numpy as np
from PIL import Image

img_Rband = Image.open("1.gif","r")
img_Gband = Image.open("2.gif","r")
img_Bband = Image.open("3.gif","r")
img_Iband = Image.open("4.gif","r")

img_r = np.array(img_Rband).flatten()
img_g = np.array(img_Gband).flatten()
img_b = np.array(img_Bband).flatten()
img_i = np.array(img_Iband).flatten()

mean_r = np.mean(img_r,axis=0)
mean_g = np.mean(img_g,axis=0)
mean_b = np.mean(img_b,axis=0)
mean_i = np.mean(img_i,axis=0)

#pre-process Data
#zero-out mean
img_r = img_r - mean_r
img_g = img_g - mean_g
img_b = img_b - mean_b
img_i = img_i - mean_i

X = np.column_stack((img_r,img_g,img_b,img_i))
mean = np.column_stack((mean_r,mean_g,mean_b,mean_i))
cov = (X.T @ X)/X.shape[0]

_,eigen_vec = np.linalg.eig(cov)

P_C = np.zeros((X.shape[1],X.shape[0]))

for a in range(4):
    P_C[a]=(np.dot(eigen_vec[:,a],X.T))

new_r = np.array(P_C[0]).reshape((512,512)).astype(np.uint8)
new_g = np.array(P_C[1]).reshape((512,512)).astype(np.uint8)
new_b = np.array(P_C[2]).reshape((512,512)).astype(np.uint8)
new_i = np.array(P_C[3]).reshape((512,512)).astype(np.uint8)

new_img_r = Image.fromarray(new_r)
new_img_g = Image.fromarray(new_g)
new_img_b = Image.fromarray(new_b)
new_img_i = Image.fromarray(new_i)

new_img_r.save("PCA_1.png")
new_img_g.save("PCA_2.png")
new_img_b.save("PCA_3.png")
new_img_i.save("PCA_4.png")

