#Author - Keshav Bimbraw

import numpy as np
from sklearn.model_selection import cross_val_score
import imageio
import glob
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time
import sys

#subject = 'Anthony'
CV_folds = [2, 5, 10]

subjects = ["Anthony", "Camren", "Kai", "Keshav", "Kevin", "Layal"]

for j in range(len(CV_folds)):
     print("Now starting CV Folds = " + str(CV_folds[j]))
     for i in range(len(subjects)):
          print("Now starting subject: " + str(subjects[i]))
          sys.stdout = open("C:/Users/bimbr/OneDrive/Desktop/SMG/data_MQP_classification/Results/" + str(subjects[i]) + "/CV/output_CV_" + str(CV_folds[j]) + "_folds.txt", 'w')

          time_start = time.perf_counter()

          image_tensor = []#np.zeros(100, 800, 640, 3)

          for im_path in glob.glob("C:/Users/bimbr/OneDrive/Desktop/SMG/data_MQP_classification/" + str(subjects[i]) + "/image*.png"):
               im = imageio.imread(im_path)
               image_tensor.append(im)

          image_tensor = np.asarray(image_tensor)

          label = []

          for i in range(0, 5):
               for j in range(0, 100):
                    label.append(i)

          label = np.asarray(label)

          image_flatten = image_tensor.reshape((500, 800*640*3))

          X_train, X_test, y_train, y_test = train_test_split(image_flatten, label, test_size=0.2)

          svclassifier = SVC(kernel='linear')
          print('Started Generating the CV score with SVC')

          accuracy = cross_val_score(svclassifier, image_flatten, label, scoring='accuracy', cv=CV_folds[j], verbose=1)
          print("The CV accuracy score is: " + str(accuracy))
          print('Cross Validation Done for CV Folds = ' + str(CV_folds[j]) + ' for subject = ' + str(subjects[i]) + '!')

          time_end = time.perf_counter()
          time_min = time_end/60
          print("Time in seconds: " + str(time_end) + ", and time in minutes: " + str(time_end))

          sys.stdout.close()