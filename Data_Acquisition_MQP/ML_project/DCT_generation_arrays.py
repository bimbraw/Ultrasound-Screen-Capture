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

image_array = np.load("P:/MQP_data/ML_project_data/data/subject_3/X_3.npy")
print(image_array.shape)
print(image_array[0].shape)

def get_2D_dct(img):
    return fftpack.dct(fftpack.dct(img.T, norm='ortho').T, norm='ortho')

dct_array = []

for i in range(0, image_array.shape[0]):
    dct1_1 = get_2D_dct(image_array[i])
    dct_values1_1 = np.abs(dct1_1[:10, :10])
    dct_array.append(dct_values1_1)
    #print(i)

dct_array = np.array(dct_array)
print(dct_array.shape)

np.save("P:/MQP_data/ML_project_data/data/subject_3/X_3_dct.npy", dct_array)

#pixels1_1 = get_image_from_url(image_url=image_url1_1, size=(640, 640))

#np.save("P:/MQP_data/ML_project_data/data/subject_3/X_3_dct.npy", subject_1_perpendicular)