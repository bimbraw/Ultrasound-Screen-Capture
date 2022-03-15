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
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from PIL import Image

time_start = time.perf_counter()
path = 'C:/Users/bimbr/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Github/medical_imaging_project/new_data/compressed_images/'
labels_path = 'C:/Users/bimbr/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Github/medical_imaging_project/' \
              'new_data/image_with_label/labels.txt'

q = []

for j in range(1, 4):
    for i in range(0, 70):
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

result_arr1 = result_arr1.reshape((140, 450, 450, 3))
result_arr2 = result_arr2.reshape((140, 450, 450, 3))
result_arr3 = result_arr3.reshape((140, 450, 450, 3))

result_arr1 = result_arr1[70:, :, :, :]
result_arr2 = result_arr2[70:, :, :, :]
result_arr3 = result_arr3[70:, :, :, :]

print(result_arr1.shape)
print(result_arr2.shape)
print(result_arr3.shape)

result_arr = np.concatenate((result_arr1, result_arr2, result_arr3))

print(result_arr.shape)

result_arr = result_arr.reshape((210, 450, 450, 3))

print(result_arr.shape)
image_flatten = result_arr.reshape((210, 450*450*3))
print(image_flatten.shape)
print(labels.shape)

X_train, X_test, y_train, y_test = train_test_split(image_flatten, labels, test_size=0.2)
print('split data')

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


svclassifier = SVC(kernel='linear', verbose=1)
print('Started Training')
svclassifier.fit(X_train, y_train)
print('Training done!')

print('Started Predictions')
y_pred = svclassifier.predict(X_test)
print('Predictions done!')

filename = 'C:/Users/bimbr/OneDrive/Desktop/PhD/Coursework/Fed/finalized_model_2.sav'
joblib.dump(svclassifier, filename)
print(f'Model saved as {filename}')

cm = confusion_matrix(y_test, y_pred)
#print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix for SVC, 3 classes')
fig.colorbar(cax)
#ax.set_xticklabels([''] + labels)
#ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(svclassifier.classes_)

time_end = time.perf_counter()

print(time_end)