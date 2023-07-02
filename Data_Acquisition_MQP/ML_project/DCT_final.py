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
image_url2_1 = 'P:/MQP_data/ML_project_data/Perpendicular_1/S_1_1/image_1_150.png'
image_url3_1 = 'P:/MQP_data/ML_project_data/Perpendicular_1/S_1_1/image_1_250.png'
image_url4_1 = 'P:/MQP_data/ML_project_data/Perpendicular_1/S_1_1/image_1_350.png'
image_url5_1 = 'P:/MQP_data/ML_project_data/Perpendicular_1/S_1_1/image_1_450.png'

image_url1_2 = 'P:/MQP_data/ML_project_data/Perpendicular_1/S_1_2/image_1_050.png'
image_url2_2 = 'P:/MQP_data/ML_project_data/Perpendicular_1/S_1_2/image_1_150.png'
image_url3_2 = 'P:/MQP_data/ML_project_data/Perpendicular_1/S_1_2/image_1_250.png'
image_url4_2 = 'P:/MQP_data/ML_project_data/Perpendicular_1/S_1_2/image_1_350.png'
image_url5_2 = 'P:/MQP_data/ML_project_data/Perpendicular_1/S_1_2/image_1_450.png'

image_url1_3 = 'P:/MQP_data/ML_project_data/Perpendicular_1/S_1_3/image_1_050.png'
image_url2_3 = 'P:/MQP_data/ML_project_data/Perpendicular_1/S_1_3/image_1_150.png'
image_url3_3 = 'P:/MQP_data/ML_project_data/Perpendicular_1/S_1_3/image_1_250.png'
image_url4_3 = 'P:/MQP_data/ML_project_data/Perpendicular_1/S_1_3/image_1_350.png'
image_url5_3 = 'P:/MQP_data/ML_project_data/Perpendicular_1/S_1_3/image_1_450.png'

def get_image_from_url(image_url, size=(640, 640)):
    image = imread(image_url)
    img = np.array(image)
    print(img.shape)
    return img

def get_2D_dct(img):
    return fftpack.dct(fftpack.dct(img.T, norm='ortho').T, norm='ortho')

pixels1_1 = get_image_from_url(image_url=image_url1_1, size=(640, 640))
pixels2_1 = get_image_from_url(image_url=image_url2_1, size=(640, 640))
pixels3_1 = get_image_from_url(image_url=image_url3_1, size=(640, 640))
pixels4_1 = get_image_from_url(image_url=image_url4_1, size=(640, 640))
pixels5_1 = get_image_from_url(image_url=image_url5_1, size=(640, 640))

pixels1_2 = get_image_from_url(image_url=image_url1_2, size=(640, 640))
pixels2_2 = get_image_from_url(image_url=image_url2_2, size=(640, 640))
pixels3_2 = get_image_from_url(image_url=image_url3_2, size=(640, 640))
pixels4_2 = get_image_from_url(image_url=image_url4_2, size=(640, 640))
pixels5_2 = get_image_from_url(image_url=image_url5_2, size=(640, 640))

pixels1_3 = get_image_from_url(image_url=image_url1_3, size=(640, 640))
pixels2_3 = get_image_from_url(image_url=image_url2_3, size=(640, 640))
pixels3_3 = get_image_from_url(image_url=image_url3_3, size=(640, 640))
pixels4_3 = get_image_from_url(image_url=image_url4_3, size=(640, 640))
pixels5_3 = get_image_from_url(image_url=image_url5_3, size=(640, 640))

dct1_1 = get_2D_dct(pixels1_1)
dct2_1 = get_2D_dct(pixels2_1)
dct3_1 = get_2D_dct(pixels3_1)
dct4_1 = get_2D_dct(pixels4_1)
dct5_1 = get_2D_dct(pixels5_1)

dct1_2 = get_2D_dct(pixels1_2)
dct2_2 = get_2D_dct(pixels2_2)
dct3_2 = get_2D_dct(pixels3_2)
dct4_2 = get_2D_dct(pixels4_2)
dct5_2 = get_2D_dct(pixels5_2)

dct1_3 = get_2D_dct(pixels1_3)
dct2_3 = get_2D_dct(pixels2_3)
dct3_3 = get_2D_dct(pixels3_3)
dct4_3 = get_2D_dct(pixels4_3)
dct5_3 = get_2D_dct(pixels5_3)

dct_values1_1 = np.abs(dct1_1[:10, :10])
dct_values2_1 = np.abs(dct2_1[:10, :10])
dct_values3_1 = np.abs(dct3_1[:10, :10])
dct_values4_1 = np.abs(dct4_1[:10, :10])
dct_values5_1 = np.abs(dct5_1[:10, :10])

dct_values1_2 = np.abs(dct1_2[:10, :10])
dct_values2_2 = np.abs(dct2_2[:10, :10])
dct_values3_2 = np.abs(dct3_2[:10, :10])
dct_values4_2 = np.abs(dct4_2[:10, :10])
dct_values5_2 = np.abs(dct5_2[:10, :10])

dct_values1_3 = np.abs(dct1_3[:10, :10])
dct_values2_3 = np.abs(dct2_3[:10, :10])
dct_values3_3 = np.abs(dct3_3[:10, :10])
dct_values4_3 = np.abs(dct4_3[:10, :10])
dct_values5_3 = np.abs(dct5_3[:10, :10])

fig, axs = plt.subplots(3, 5)
axs[0, 0].matshow(dct_values1_1, cmap=plt.cm.Paired)
axs[0, 1].matshow(dct_values2_1, cmap=plt.cm.Paired)
axs[0, 2].matshow(dct_values3_1, cmap=plt.cm.Paired)
axs[0, 3].matshow(dct_values4_1, cmap=plt.cm.Paired)
axs[0, 4].matshow(dct_values5_1, cmap=plt.cm.Paired)

axs[1, 0].matshow(dct_values1_2, cmap=plt.cm.Paired)
axs[1, 1].matshow(dct_values2_2, cmap=plt.cm.Paired)
axs[1, 2].matshow(dct_values3_2, cmap=plt.cm.Paired)
axs[1, 3].matshow(dct_values4_2, cmap=plt.cm.Paired)
axs[1, 4].matshow(dct_values5_2, cmap=plt.cm.Paired)

axs[2, 0].matshow(dct_values1_3, cmap=plt.cm.Paired)
axs[2, 1].matshow(dct_values2_3, cmap=plt.cm.Paired)
axs[2, 2].matshow(dct_values3_3, cmap=plt.cm.Paired)
axs[2, 3].matshow(dct_values4_3, cmap=plt.cm.Paired)
axs[2, 4].matshow(dct_values5_3, cmap=plt.cm.Paired)

plt.show()