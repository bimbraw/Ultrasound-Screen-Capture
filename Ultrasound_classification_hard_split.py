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

time_start = time.perf_counter()
image_tensor = []#np.zeros(100, 800, 640, 3)
subject = 'Anthony'

for im_path in glob.glob("C:/Users/bimbr/OneDrive/Desktop/SMG/data_MQP_classification_parallel_config/" + str(subject) + "/image*.png"):
     im = imageio.imread(im_path)
     #plt.imshow(im, interpolation='nearest')
     #plt.show()
     image_tensor.append(im)

image_tensor = np.asarray(image_tensor)

print(im.shape)
print(type(image_tensor))
print(image_tensor.shape)

label = []

for i in range(0, 5):
     for j in range(0, 100):
          label.append(i)

label = np.asarray(label)
print(label)
print(label.shape)

image_flatten = image_tensor.reshape((500, 800*640*3))
print(image_flatten.shape)

X_train_1 = image_flatten[:80, :]
X_train_2 = image_flatten[100:180, :]
X_train_3 = image_flatten[200:280, :]
X_train_4 = image_flatten[300:380, :]
X_train_5 = image_flatten[400:480, :]
X_test_1 = image_flatten[80:100, :]
X_test_2 = image_flatten[180:200, :]
X_test_3 = image_flatten[280:300, :]
X_test_4 = image_flatten[380:400, :]
X_test_5 = image_flatten[480:500, :]
y_train_1 = label[:80]
y_train_2 = label[100:180]
y_train_3 = label[200:280]
y_train_4 = label[300:380]
y_train_5 = label[400:480]
y_test_1 = label[80:100]
y_test_2 = label[180:200]
y_test_3 = label[280:300]
y_test_4 = label[380:400]
y_test_5 = label[480:500]

print(X_train_3.shape)
print(X_test_3.shape)
print(y_train_3.shape)
print(y_test_3.shape)

X_train = np.concatenate((X_train_1, X_train_2, X_train_3, X_train_4, X_train_5))
X_test = np.concatenate((X_test_1, X_test_2, X_test_3, X_test_4, X_test_5))
y_train = np.concatenate((y_train_1, y_train_2, y_train_3, y_train_4, y_train_5))
y_test = np.concatenate((y_test_1, y_test_2, y_test_3, y_test_4, y_test_5))
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

svclassifier = SVC(kernel='linear', verbose=1)
print('Started Training')
svclassifier.fit(X_train, y_train)
print('Training done!')

filename = 'C:/Users/bimbr/OneDrive/Desktop/SMG/data_MQP_classification_parallel_config/' + str(subject) + '/finalized_model.sav'
joblib.dump(svclassifier, filename)
print(f'Model saved as {filename}')

print('Started Predictions')
y_pred = svclassifier.predict(X_test)
print('Predictions done!')

cm = confusion_matrix(y_test, y_pred)
#print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix for SVC, 5 classes')
fig.colorbar(cax)
#ax.set_xticklabels([''] + labels)
#ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
plt.savefig(str(subject) + '5_states.png')

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(svclassifier.classes_)
np.save('C:/Users/bimbr/OneDrive/Desktop/SMG/data_MQP_classification_parallel_config/' + str(subject) + '/class_values', svclassifier.classes_)

time_end = time.perf_counter()

print(time_end)