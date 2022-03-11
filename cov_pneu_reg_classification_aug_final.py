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

num_val = 3
case = 'local_' + str(num_val)


time_start = time.perf_counter()
path = 'C:/Users/bimbr/OneDrive/Desktop/PhD/Coursework/Fed/final_data/np_files/'

X = np.load(path + case + '.npy')
y = np.load(path + 'labels_' + str(num_val) + '.npy')

print(X.shape)
print(y.shape)

X = X.reshape((4800, 450*450*3))

ts = 0.2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts)
print('split data')

chosen_len = 105

X_train = X_train[:int((1-ts)*chosen_len),:]
y_train = y_train[:int((1-ts)*chosen_len),]
X_test = X_test[:int((ts)*chosen_len),:]
y_test = y_test[:int((ts)*chosen_len),]

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

svclassifier = SVC(kernel='linear', verbose=1)
print('Started Training')
svclassifier.fit(X_train, y_train)
print('Training done!')

filename = path + 'Results/'+ case + '_model.sav'
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
plt.title('Confusion matrix for ' + case)
fig.colorbar(cax)
#ax.set_xticklabels([''] + labels)
#ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(path + 'Results/' + case + '.png')
plt.show()

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(svclassifier.classes_)

accuracy_val = accuracy_score(y_pred, y_test)
np.save(path + 'Results/' + 'accuracy_' + case, accuracy_val)

time_end = time.perf_counter()

print(time_end)