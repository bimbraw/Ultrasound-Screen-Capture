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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import figure
import addcopyfighandler

time_start = time.perf_counter()

im1 = mpimg.imread("C:/Users/bimbr/OneDrive/Desktop/SMG/data_MQP_classification/Keshav/image60.png")
im2 = mpimg.imread("C:/Users/bimbr/OneDrive/Desktop/SMG/data_MQP_classification/Keshav/image160.png")
im3 = mpimg.imread("C:/Users/bimbr/OneDrive/Desktop/SMG/data_MQP_classification/Keshav/image260.png")
im4 = mpimg.imread("C:/Users/bimbr/OneDrive/Desktop/SMG/data_MQP_classification/Keshav/image360.png")
im5 = mpimg.imread("C:/Users/bimbr/OneDrive/Desktop/SMG/data_MQP_classification/Keshav/image460.png")

im_init = np.zeros((800, 640, 3))

val_init_vals = [50, 150, 250, 350, 450]

val_init = val_init_vals[4]
samples = 15
for i in range(0, samples):
    val = val_init + i
    str_name = "C:/Users/bimbr/OneDrive/Desktop/SMG/data_MQP_classification/Keshav/image" + str(val) + ".png"
    im_agg = mpimg.imread(str_name)
    im_agg = im_init + im_agg
    im_init = im_agg

print(im_agg.size)

plt.figure(figsize=(5, 5))
imgplot = plt.imshow(im_agg/samples)

plt.show()

'''
im_new = im5 - im5

plt.figure(figsize=(5, 5))
imgplot = plt.imshow(im_new * 2)
plt.show()
'''
#fig = plt.figure(figsize=(12, 10))
#plt.show()

'''
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
plt.savefig(str(subject) + '5_states.png')

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(svclassifier.classes_)
np.save('C:/Users/bimbr/OneDrive/Desktop/SMG/data_MQP_classification_parallel_config/' + str(subject) + '/class_values', svclassifier.classes_)

time_end = time.perf_counter()

print(time_end)
'''