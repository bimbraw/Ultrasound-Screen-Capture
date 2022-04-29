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
from scipy import ndimage

time_start = time.perf_counter()
image_tensor = []#np.zeros(100, 800, 640, 3)

#15 minutes per subject

rounds = 6
classes = 5
len_classes = 100
configurations = ["Perpendicular_1", "Perpendicular_2", "Perpendicular_3", "Parallel_upwards", "Parallel_downwards"]
configuration = configurations[0]
subjects = ["Keshav", "Anthony", "Camren", "Layal", "Kevin"]
subject = subjects[1]

for im_path in glob.glob("P:/MQP_data/" + str(configuration) + "/" + str(subject) + "/image*.png"):
     print(im_path)
     im = imageio.imread(im_path)
     im = ndimage.interpolation.zoom(im, .25)
     #plt.imshow(im, interpolation='nearest')
     #plt.show()
     image_tensor.append(im)


image_tensor = np.asarray(image_tensor)

print(im.shape)
print(type(image_tensor))
print(image_tensor.shape)

label = []

for k in range(0, 6):
     for i in range(0, 5):
          for j in range(0, 100):
               label.append(i)

label = np.asarray(label)
print(label)
print(label.shape)

image_flatten = image_tensor.reshape((3000, 640*640))
print(image_flatten.shape)

X_train, X_test, y_train, y_test = train_test_split(image_flatten, label, test_size=0.2)
print('split data')

svclassifier = SVC(kernel='linear', verbose=1)
print('Started Training')
svclassifier.fit(X_train, y_train)
print('Training done!')

filename = "P:/MQP_data/" + str(configuration) + "/Results/" + str(subject) + "/finalized_model.sav"
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
#plt.show()
plt.savefig("P:/MQP_data/" + str(configuration) + "/Results/" + str(subject) + "5_states.png")

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(svclassifier.classes_)
np.save("P:/MQP_data/" + str(configuration) + "/Results/" + str(subject) + "/class_values", svclassifier.classes_)

time_end = time.perf_counter()

print(time_end)