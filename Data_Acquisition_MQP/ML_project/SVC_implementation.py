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

X = np.load("P:/MQP_data/ML_project_data/data/subject_1/X_1_dct.npy")
print(X.shape)
y = np.load("P:/MQP_data/ML_project_data/data/subject_1/y_1.npy")
print(y.shape)

X = X.reshape((9000, 100))
print(X.shape)

#1/3 test-train split
X_train = X[:6000, :]
y_train = y[:6000]
X_test = X[6000:, :]
y_test = y[6000:]

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print('split data')

svclassifier = SVC(kernel='linear', verbose=1)
print('Started Training')
svclassifier.fit(X_train, y_train)
print('Training done!')

#filename = "P:/MQP_data/" + str(configuration) + "/Final_Results/" + str(subject) + "/finalized_model_downsampled.sav"
#joblib.dump(svclassifier, filename)
#print(f'Model saved as {filename}')

print('Started Predictions')
y_pred = svclassifier.predict(X_test)
print('Predictions done!')

cm = confusion_matrix(y_test, y_pred)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix for SVC, 5 classes')
fig.colorbar(cax)
plt.xlabel('Predicted')
plt.ylabel('True')
#plt.savefig("P:/MQP_data/" + str(configuration) + "/Final_Results/" + str(subject) + "/5_states_downsampled.png")
plt.show()

# with open("P:/MQP_data/" + str(configuration) + "/Final_Results/" + str(subject) + "/5_states_downsampled.txt", "a") as external_file:
#     print(confusion_matrix(y_test,y_pred), file=external_file)
#     print(classification_report(y_test,y_pred), file=external_file)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(svclassifier.classes_)
#np.save("P:/MQP_data/" + str(configuration) + "/Final_Results/" + str(subject) + "/class_values_downsampled", svclassifier.classes_)
