import cv2
import numpy as np
import os
import pyautogui
#from moviepy.editor import VideoFileClip
#from moviepy.video.fx.crop import crop
import time
import imageio
import glob
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from PIL import Image
import random

time_start = time.perf_counter()
path = 'C:/Users/bimbr/OneDrive/Desktop/PhD/Coursework/Fed/final_data/'

q = []

total_per_class = 6400
samp_val = int(total_per_class/4)

for j in range(1, 4):
    for i in range(0, samp_val):
        q.append(j)

labels = np.asarray(q)

path1 = path + 'cov/'
path2 = path + 'pneu/'
path3 = path + 'reg/'

image_list1 = []
for filename in glob.glob(str(path1) + '*.jpg'): #assuming gif
    im = Image.open(filename)
    im = np.array(im)
    image_list1.append(im)

result_arr1 = np.concatenate(image_list1)
result_arr1 = result_arr1.reshape((int(result_arr1.shape[0]/450), 450, 450, 3))
result_arr1 = result_arr1[:total_per_class, :, :, :]
print(result_arr1.shape)

image_list2 = []
for filename in glob.glob(str(path2) + '*.jpg'): #assuming gif
    im = Image.open(filename)
    im = np.array(im)
    image_list2.append(im)

result_arr2 = np.concatenate(image_list1)
result_arr2 = result_arr2.reshape((int(result_arr2.shape[0]/450), 450, 450, 3))
result_arr2 = result_arr2[:total_per_class, :, :, :]
print(result_arr2.shape)

image_list3 = []
for filename in glob.glob(str(path3) + '*.jpg'): #assuming gif
    im = Image.open(filename)
    im = np.array(im)
    image_list3.append(im)

result_arr3 = np.concatenate(image_list3)
result_arr3 = result_arr3.reshape((int(result_arr3.shape[0]/450), 450, 450, 3))
result_arr3 = result_arr3[:total_per_class, :, :, :]
print(result_arr3.shape)

print(result_arr1.shape)
print(result_arr2.shape)
print(result_arr3.shape)

data_global_cov = result_arr1[:samp_val, :]
data_global_pne = result_arr2[:samp_val, :]
data_global_reg = result_arr3[:samp_val, :]

local_1_cov = result_arr1[samp_val:samp_val*2, :]
local_1_pne = result_arr2[samp_val:samp_val*2, :]
local_1_reg = result_arr3[samp_val:samp_val*2, :]

local_2_cov = result_arr1[samp_val*2:samp_val*3, :]
local_2_pne = result_arr2[samp_val*2:samp_val*3, :]
local_2_reg = result_arr3[samp_val*2:samp_val*3, :]

local_3_cov = result_arr1[samp_val*3:samp_val*4, :]
local_3_pne = result_arr2[samp_val*3:samp_val*4, :]
local_3_reg = result_arr3[samp_val*3:samp_val*4, :]

print(data_global_cov.shape)
print(data_global_pne.shape)
print(data_global_reg.shape)

print(local_1_cov.shape)
print(local_1_pne.shape)
print(local_1_reg.shape)

print(local_2_cov.shape)
print(local_2_pne.shape)
print(local_2_reg.shape)

print(local_3_cov.shape)
print(local_3_pne.shape)
print(local_3_reg.shape)

data_global = np.concatenate((data_global_cov,
                              data_global_pne,
                              data_global_reg))

local_1 = np.concatenate((local_1_cov,
                          local_1_pne,
                          local_1_reg))

local_2 = np.concatenate((local_2_cov,
                          local_2_pne,
                          local_2_reg))

local_3 = np.concatenate((local_3_cov,
                          local_3_pne,
                          local_3_reg))

print(data_global.shape)
print(local_1.shape)
print(local_2.shape)
print(local_3.shape)

np.save('C:/Users/bimbr/OneDrive/Desktop/PhD/Coursework/Fed/final_data/np_files/data_global.npy', data_global)
np.save('C:/Users/bimbr/OneDrive/Desktop/PhD/Coursework/Fed/final_data/np_files/labels_global.npy', labels)
np.save('C:/Users/bimbr/OneDrive/Desktop/PhD/Coursework/Fed/final_data/np_files/local_1.npy', local_1)
np.save('C:/Users/bimbr/OneDrive/Desktop/PhD/Coursework/Fed/final_data/np_files/labels_1.npy', labels)
np.save('C:/Users/bimbr/OneDrive/Desktop/PhD/Coursework/Fed/final_data/np_files/local_2.npy', local_2)
np.save('C:/Users/bimbr/OneDrive/Desktop/PhD/Coursework/Fed/final_data/np_files/labels_2.npy', labels)
np.save('C:/Users/bimbr/OneDrive/Desktop/PhD/Coursework/Fed/final_data/np_files/local_3.npy', local_3)
np.save('C:/Users/bimbr/OneDrive/Desktop/PhD/Coursework/Fed/final_data/np_files/labels_3.npy', labels)

'''
result_arr = result_arr.reshape((420, 450, 450, 3))

print(result_arr.shape)
image_flatten = result_arr.reshape((420, 450*450*3))
print(image_flatten.shape)
print(labels.shape)

result_arr1 = result_arr1.reshape((140, 450, 450, 3))
result_arr2 = result_arr2.reshape((140, 450, 450, 3))
result_arr3 = result_arr3.reshape((140, 450, 450, 3))

result_arr1 = result_arr1[:35, :, :, :]
result_arr2 = result_arr2[:35, :, :, :]
result_arr3 = result_arr3[:35, :, :, :]

print(result_arr1.shape)
print(result_arr2.shape)
print(result_arr3.shape)

result_arr = np.concatenate((result_arr1, result_arr2, result_arr3))
'''
time_end = time.perf_counter()

print(time_end)