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

from tensorflow import keras
from keras_preprocessing.image import img_to_array, load_img

# Initialising the ImageDataGenerator class.
# We will pass in the augmentation parameters in the constructor.
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.5, 1.5))

type = 'reg'
if type == 'reg':
    max_range = 10
    number = 7000/max_range
elif type == 'cov':
    max_range = 5
    number = 7000/max_range
elif type == 'pneu':
    max_range = 14
    number = 7000/max_range

for i in range(1, max_range + 1):
    specific_name = str(type) + '_' + str(i) + ' resized'
    # Loading a sample image
    img = load_img('C:/Users/bimbr/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Github/medical_imaging_project/new_data/compressed_images/Resized_images/' + str(specific_name) + '.jpg')
    # Converting the input sample image to an array
    x = img_to_array(img)
    # Reshaping the input image
    x = x.reshape((1,) + x.shape)

    # Generating and saving 5 augmented samples
    # using the above defined parameters.
    i = 0
    for batch in datagen.flow(x, batch_size=5,
                              save_to_dir='C:/Users/bimbr/OneDrive/Desktop/PhD/Coursework/Fed/final_data/' +
                                          str(type) + '/',
                              save_prefix = str(specific_name), save_format='jpg'):
        i += 1
        if i > number:
            break
'''
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

#my code for resizing the image
'''
'''
dirs = os.listdir(path)

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((450,450), Image.ANTIALIAS)
            imResize.save(f + ' resized.jpg', 'png', quality=90)

resize()
'''
'''
image_tensor = []#np.zeros(100, 800, 640, 3)
subject = 'Camren'

for im_path in glob.glob("C:/Users/bimbr/OneDrive/Desktop/SMG/data_MQP_classification/" + str(subject) + "/image*.png"):
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

filename = 'C:/Users/bimbr/OneDrive/Desktop/SMG/data_MQP_classification/' + str(subject) + '/finalized_model.sav'
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
np.save('C:/Users/bimbr/OneDrive/Desktop/SMG/data_MQP_classification/' + str(subject) + '/class_values', svclassifier.classes_)
'''
time_end = time.perf_counter()

print(time_end)