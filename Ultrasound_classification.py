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
subject = 'Camren'

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

X_train, X_test, y_train, y_test = train_test_split(image_flatten, label, test_size=0.2)
print('split data')

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
plt.savefig(str(subject) + '6_states.png')

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(svclassifier.classes_)
np.save('C:/Users/bimbr/OneDrive/Desktop/SMG/data_MQP_classification_parallel_config/' + str(subject) + '/class_values', svclassifier.classes_)

time_end = time.perf_counter()

print(time_end)