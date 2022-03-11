import cv2
import numpy as np
import os
import pyautogui
#from moviepy.editor import VideoFileClip
#from moviepy.video.fx.crop import crop
import joblib
import imageio
import glob
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

filename = 'C:/Users/bimbr/OneDrive/Desktop/SMG/data_MQP_classification/Camren/finalized_model.sav'
print('Loading model')
loaded_model = joblib.load(filename)
print('Loaded the model')

while True:
     image = pyautogui.screenshot(region=(640, 0, 640, 800))
     image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
     cv2.imwrite("C:/Users/bimbr/OneDrive/Desktop/SMG/data_MQP_classification/Camren/prediction_test_data.png", image)
     image_flatten = image.reshape((1, 800 * 640 * 3))
     y_pred = loaded_model.predict(image_flatten)
     print(y_pred)