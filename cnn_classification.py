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
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop, Adam
import visualkeras

start = time.time()
print(tf.__version__)
print('All imports successful!') #if this works then everything is installed well
tf.compat.v1.random.set_random_seed(1234)

model = Sequential()
#
model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (800, 640, 3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#
model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
# fully connected
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation = "softmax"))

# Define the optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print(model.summary())

visualkeras.layered_view(model).show()
epochs = 10  # for better result increase the epochs
batch_size = 250

image_tensor = []#np.zeros(100, 800, 640, 3)

for im_path in glob.glob("classification_test/*.png"):
     im = imageio.imread(im_path)
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

X_train, X_test, Y_train, Y_test = train_test_split(image_flatten, label, test_size=0.2)
print('split data')

#X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)


X_train = np.reshape(X_train, (X_train.shape[0], 800, 640, 3))
X_test = np.reshape(X_test, (X_test.shape[0], 800, 640, 3))
#X_val = np.reshape(X_val, (X_val.shape[0], 800, 640, 3))
print("X_train shape: ", X_train.shape)
print("X_train shape: ", X_test.shape)
#print("X_val shape: ", X_val.shape)
print("Y_train shape: ", Y_train.shape)
print("Y_train shape: ", Y_test.shape)
#print("Y_val shape: ", Y_val.shape)

model.fit(X_train, Y_train, validation_split=0.1, epochs=25)

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