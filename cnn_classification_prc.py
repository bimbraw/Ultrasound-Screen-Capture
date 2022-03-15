import numpy as np
#get_ipython().magic(u'matplotlib qt')
import tensorflow as tf
import time
import imageio
import glob
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv3D, MaxPooling2D, Flatten, Input, Activation, BatchNormalization, Dropout, Reshape
from tensorflow.keras.optimizers import RMSprop, Adam
import visualkeras
from PIL import Image

start = time.time()
print(tf.__version__)
print('All imports successful!') #if this works then everything is installed well
tf.compat.v1.random.set_random_seed(1234)

n_filters = 8
kernel_dim = 2
activation_val = 'softmax'

model = Sequential()
#
model.add(Conv2D(filters = n_filters, kernel_size = (kernel_dim,kernel_dim),padding = 'Same',
                 activation =activation_val, input_shape = (450, 450, 3)))
model.add(Conv2D(filters = n_filters, kernel_size = (kernel_dim,kernel_dim),padding = 'Same',
                 activation =activation_val))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#
model.add(Conv2D(filters = n_filters, kernel_size = (kernel_dim,kernel_dim),padding = 'Same',
                 activation =activation_val))
model.add(Conv2D(filters = n_filters, kernel_size = (kernel_dim,kernel_dim),padding = 'Same',
                 activation =activation_val))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#
model.add(Conv2D(filters = n_filters, kernel_size = (kernel_dim,kernel_dim),padding = 'Same',
                 activation =activation_val))
model.add(Conv2D(filters = n_filters, kernel_size = (kernel_dim,kernel_dim),padding = 'Same',
                 activation =activation_val))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#
model.add(Conv2D(filters = n_filters, kernel_size = (kernel_dim,kernel_dim),padding = 'Same',
                 activation =activation_val))
model.add(Conv2D(filters = n_filters, kernel_size = (kernel_dim,kernel_dim),padding = 'Same',
                 activation =activation_val))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# fully connected
model.add(Flatten())
model.add(Dense(16, activation = activation_val))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(3, activation = activation_val))

# Define the optimizer
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print(model.summary())

visualkeras.layered_view(model).show()

path = 'C:/Users/bimbr/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Github/medical_imaging_project/new_data/compressed_images/'
labels_path = 'C:/Users/bimbr/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Github/medical_imaging_project/' \
              'new_data/image_with_label/labels.txt'

q = []

for j in range(1, 4):
    for i in range(0, 140):
        q.append(j)

labels = np.asarray(q)

path1 = path + 'covid/'
path2 = path + 'pneu/'
path3 = path + 'reg/'

image_list1 = []
for filename in glob.glob(str(path1) + '*.jpg'): #assuming gif
    im = Image.open(filename)
    im = np.array(im)
    image_list1.append(im)

result_arr1 = np.concatenate(image_list1)

image_list2 = []
for filename in glob.glob(str(path2) + '*.jpg'): #assuming gif
    im = Image.open(filename)
    im = np.array(im)
    image_list2.append(im)

result_arr2 = np.concatenate(image_list2)

image_list3 = []
for filename in glob.glob(str(path3) + '*.jpg'): #assuming gif
    im = Image.open(filename)
    im = np.array(im)
    image_list3.append(im)

result_arr3 = np.concatenate(image_list3)

print(result_arr1.shape)
print(result_arr2.shape)
print(result_arr3.shape)

result_arr = np.concatenate((result_arr1, result_arr2, result_arr3))

print(result_arr.shape)

result_arr = result_arr.reshape((420, 450, 450, 3))

print(result_arr.shape)
image_flatten = result_arr.reshape((420, 450*450*3))
print(image_flatten.shape)
print(labels.shape)

X_train, X_test, Y_train, Y_test = train_test_split(image_flatten, labels, test_size=0.2)
print('split data')

#X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)


X_train = np.reshape(X_train, (X_train.shape[0], 450, 450, 3))
X_test = np.reshape(X_test, (X_test.shape[0], 450, 450, 3))
#X_val = np.reshape(X_val, (X_val.shape[0], 800, 640, 3))
print("X_train shape: ", X_train.shape)
print("X_train shape: ", X_test.shape)
#print("X_val shape: ", X_val.shape)
print("Y_train shape: ", Y_train.shape)
print("Y_train shape: ", Y_test.shape)
#print("Y_val shape: ", Y_val.shape)

model.fit(X_train, Y_train, validation_split=0.1, epochs=10)

Y_pred = model.predict(X_test)
np.save('Y_pred.npy', Y_pred)
np.save('Y_test.npy', Y_test)
'''
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
Y_pred = np.load('Y_pred.npy')
Y_test = np.load('Y_test.npy')
print(Y_pred.shape)
print(Y_test.shape)
print(Y_pred)
print(Y_test)

cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4])
disp.plot()

plt.show()
'''