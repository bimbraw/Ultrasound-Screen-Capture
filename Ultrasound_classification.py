import cv2
import numpy as np
import os
import pyautogui
#from moviepy.editor import VideoFileClip
#from moviepy.video.fx.crop import crop

import imageio
import glob
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

image_tensor = []#np.zeros(100, 800, 640, 3)

for im_path in glob.glob("classification_test/*.png"):
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

svclassifier = SVC(kernel='linear')
print('Started Training')
svclassifier.fit(X_train, y_train)
print('Training done!')

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
plt.savefig('6_states.png')

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))