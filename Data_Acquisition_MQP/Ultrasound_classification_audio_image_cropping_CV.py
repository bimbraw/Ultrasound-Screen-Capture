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

#15 minutes per subject

rounds = 6
classes = 5
len_classes = 100
configurations = ["Perpendicular_1", "Perpendicular_2", "Perpendicular_3", "Parallel_upwards", "Parallel_downwards"]
#configuration = configurations[0]
subjects = ["Keshav", "Anthony", "Camren_n", "Layal_n", "Kevin"]
subject = subjects[2]

qqqq = 1

configuration = configurations[int(qqqqq)]
image_tensor = []  # np.zeros(100, 800, 640, 3)

for im_path in glob.glob("P:/MQP_data/" + str(configuration) + "/" + str(subject) + "/image*.png"):
     print(im_path)
     im = imageio.imread(im_path)
     im = im[50:450, 40:600]
     im = ndimage.interpolation.zoom(im, .25)
     #plt.imshow(im, interpolation='nearest')
     #plt.show()
     image_tensor.append(im)

image_tensor = np.asarray(image_tensor)
#image_tensor = image_tensor[1000:, :, :]
print(im.shape)
print(type(image_tensor))
print(image_tensor.shape)

#75 * 140
#im = im[50:300, 40:600]

#plt.imshow(im, interpolation='nearest')
#plt.show()

label = []

for k in range(0, 6):
     for i in range(0, 5):
          for j in range(0, 100):
               label.append(i)

label = np.asarray(label)
print(label)
print(label.shape)

image_flatten = image_tensor.reshape((3000, im.shape[0]*im.shape[1]))
print(image_flatten.shape)

label_train = label[0:2500]
label_test = label[2500:]

label_train = label_train.reshape((2500, 1))
label_test = label_test.reshape((500, 1))

print(label_train.shape)
print(label_test.shape)

image_flatten_train = image_flatten[0:2500, :]
image_flatten_test = image_flatten[2500:, :]

print(image_flatten_train.shape)
print(image_flatten_test.shape)

total_array_train = np.concatenate((image_flatten_train, label_train), axis=1)
total_array_test = np.concatenate((image_flatten_test, label_test), axis=1)

print(total_array_train.shape)
print(total_array_test.shape)

X_train = total_array_train[:, :im.shape[0]*im.shape[1]]
X_test = total_array_test[:, :im.shape[0]*im.shape[1]]
y_train = total_array_train[:, im.shape[0]*im.shape[1]]
y_test = total_array_test[:, im.shape[0]*im.shape[1]]

#print(y_train.shape, y_test.shape)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print(y_train)
print(y_test)

#X_train, X_test, y_train, y_test = train_test_split(image_flatten, label, test_size=0.2)
print('split data')

svclassifier = SVC(kernel='linear', verbose=1)
print('Started Training')
svclassifier.fit(X_train, y_train)
print('Training done!')

filename = "P:/MQP_data/" + str(configuration) + "/Final_Results/" + str(subject) + "/finalized_model_downsampled.sav"
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
plt.savefig("P:/MQP_data/" + str(configuration) + "/Final_Results/" + str(subject) + "/5_states_downsampled.png")

with open("P:/MQP_data/" + str(configuration) + "/Final_Results/" + str(subject) + "/5_states_downsampled.txt", "a") as external_file:
    print(confusion_matrix(y_test,y_pred), file=external_file)
    print(classification_report(y_test,y_pred), file=external_file)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(svclassifier.classes_)
np.save("P:/MQP_data/" + str(configuration) + "/Final_Results/" + str(subject) + "/class_values_downsampled", svclassifier.classes_)

time_end = time.perf_counter()

print(time_end)