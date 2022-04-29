import io
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from urllib.request import urlopen
#import IPython
from skimage.io import imread
from scipy import ndimage
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import keras
from keras import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

X = np.load("P:/MQP_data/ML_project_data/data/subject_1/X_1_dct.npy")
print(X.shape)
y = np.load("P:/MQP_data/ML_project_data/data/subject_1/y_1.npy")
print(y.shape)

X = X.reshape((9000, 100))
print(X.shape)

#one hot encoding
y = y - 1

acc_per_fold = []
loss_per_fold = []

train_per_fold = []

# Define the K-fold Cross Validator
kfold = KFold(n_splits=10, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(X, y):
    classifier = Sequential()
    # First Hidden Layer
    classifier.add(Dense(64, activation='relu', input_dim=100))
    # Second  Hidden Layer
    classifier.add(Dense(16, activation='relu'))
    # Output Layer
    classifier.add(Dense(5, activation='softmax'))

    # Compiling the neural network
    opt = tf.optimizers.Adam(learning_rate=0.001)
    classifier.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history = classifier.fit(X[train], y[train], batch_size=4, epochs=50, verbose=0)
    eval_model = classifier.evaluate(X[train], y[train])
    print('The training accuracy is: ' + str(eval_model[1]*100))
    train_per_fold.append(eval_model[1] * 100)

    # Generate generalization metrics
    scores = classifier.evaluate(X[test], y[test], verbose=0)
    print('The testing accuracy is: ' + str(scores[1]*100))
    print(f'Score for fold {fold_no}: {classifier.metrics_names[0]} of {scores[0]}; {classifier.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1

average_accuracy_test = sum(acc_per_fold)/10
print('Average test accuracy is ' + str(average_accuracy_test) + '%')

average_accuracy_train = sum(train_per_fold)/10
print('Average train accuracy is ' + str(average_accuracy_train) + '%')
