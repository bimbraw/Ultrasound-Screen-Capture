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

image_url1_1 = 'P:/MQP_data/ML_project_data/Perpendicular_1/S_1_1/image_1_050.png'

def get_image_from_url(image_url, size=(640, 640)):
    image = imread(image_url)
    img = np.array(image)
    print(img.shape)
    return img

def get_2D_dct(img):
    return fftpack.dct(fftpack.dct(img.T, norm='ortho').T, norm='ortho')

pixels1_1 = get_image_from_url(image_url=image_url1_1, size=(640, 640))
dct1_1 = get_2D_dct(pixels1_1)
dct_values1_1 = np.abs(dct1_1[:10, :10])

#np.save("P:/MQP_data/ML_project_data/data/subject_3/X_3_dct.npy", subject_1_perpendicular)
