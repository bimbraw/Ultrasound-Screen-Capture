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

time_start = time.perf_counter()

path = 'C:/Users/bimbr/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Github/medical_imaging_project/new_data/compressed_images/'

q = []

for j in range(1, 4):
    for i in range(0, 140):
        q.append(j)

labels = np.asarray(q)

path1 = path + 'covid/'
path2 = path + 'pneu/'
path3 = path + 'reg/'

image_list1 = []
for filename in glob.glob(str(path1) + '*.jpg'): #assuming gif
    im = Image.open(filename)
    im = np.array(im)
    image_list1.append(im)

result_arr1 = np.concatenate(image_list1)

image_list2 = []
for filename in glob.glob(str(path2) + '*.jpg'): #assuming gif
    im = Image.open(filename)
    im = np.array(im)
    image_list2.append(im)

result_arr2 = np.concatenate(image_list2)

image_list3 = []
for filename in glob.glob(str(path3) + '*.jpg'): #assuming gif
    im = Image.open(filename)
    im = np.array(im)
    image_list3.append(im)

result_arr3 = np.concatenate(image_list3)

print(result_arr1.shape)
print(result_arr2.shape)
print(result_arr3.shape)

result_arr = np.concatenate((result_arr1, result_arr2, result_arr3))

print(result_arr.shape)

result_arr = result_arr.reshape((420, 450, 450, 3))

print(result_arr.shape)
image_flatten = result_arr.reshape((420, 450*450*3))
print(image_flatten.shape)
print(labels.shape)

X_train, X_test, y_train, y_test = train_test_split(image_flatten, labels, test_size=0.2)
print('split data')

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

filename1 = 'C:/Users/bimbr/OneDrive/Desktop/PhD/Coursework/Fed/finalized_model_1.sav'
model1 = joblib.load(filename1)
print(f'Model loaded {filename1}')
filename2 = 'C:/Users/bimbr/OneDrive/Desktop/PhD/Coursework/Fed/finalized_model_2.sav'
model2 = joblib.load(filename2)
print(f'Model loaded {filename2}')

#generating predictions for the first model
y_pred1 = model1.predict(X_test)
#generating predictions for the second model
y_pred2 = model2.predict(X_test)

cm = confusion_matrix(y_test, y_pred1)
#print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix for SVC, 3 classes model 1')
fig.colorbar(cax)
#ax.set_xticklabels([''] + labels)
#ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print(classification_report(y_test,y_pred1))
accuracy1 = accuracy_score(y_test, y_pred1)
print(f'The accuracy score for model1 is: {accuracy1}')

cm = confusion_matrix(y_test, y_pred2)
#print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix for SVC, 3 classes model 2')
fig.colorbar(cax)
#ax.set_xticklabels([''] + labels)
#ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print(classification_report(y_test,y_pred2))
accuracy2 = accuracy_score(y_test, y_pred2)
print(f'The accuracy score for model1 is: {accuracy2}')

time_end = time.perf_counter()

print(time_end)