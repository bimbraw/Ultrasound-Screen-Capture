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

# [Source](http://bugra.github.io/work/notes/2014-07-12/discre-fourier-cosine-transform-dft-dct-image-compression/)
# [My Notes](https://docs.google.com/document/d/1yIDTEsXFkLLV5sLmPHtV8-X3GZ2Oa-X2zYvS1BaXMpw/edit?usp=sharing)

# image_url='http://i.imgur.com/8vuLtqi.png'
image_url = 'P:/MQP_data/ML_project_data/Perpendicular_1/S_1_1/image_1_450.png'


# reads the image from url using PIL and converting it into a numpy array after converted grayscale image.
def get_image_from_url(image_url='P:/MQP_data/ML_project_data/Perpendicular_1/S_1_1/image_1_450.png', size=(64, 64)):
    image = imread(image_url)
    #image_file = io.BytesIO(file_descriptor.read())
    #image = Image.open(image_file)
    #img_color = image.resize(size, 1)
    #img_grey = img_color.convert('L')
    image = ndimage.interpolation.zoom(image, .1)
    img = np.array(image, dtype=np.float)
    print(img.shape)
    return img


def get_2D_dct(img):
    """ Get 2D Cosine Transform of Image
    """
    return fftpack.dct(fftpack.dct(img.T, norm='ortho').T, norm='ortho')


def get_2d_idct(coefficients):
    """ Get 2D Inverse Cosine Transform of Image
    """
    return fftpack.idct(fftpack.idct(coefficients.T, norm='ortho').T, norm='ortho')


def get_reconstructed_image(raw):
    img = raw.clip(0, 255)
    img = img.astype('uint8')
    img = Image.fromarray(img)
    return img


pixels = get_image_from_url(image_url=image_url, size=(64, 64))
dct_size = pixels.shape[0]
dct = get_2D_dct(pixels)
reconstructed_images = []

for ii in range(dct_size):
    dct_copy = dct.copy()
    dct_copy[ii:, :] = 0
    dct_copy[:, ii:] = 0

    # Reconstructed image
    r_img = get_2d_idct(dct_copy)
    reconstructed_image = get_reconstructed_image(r_img)

    # Create a list of images
    reconstructed_images.append(reconstructed_image)

plt.figure(figsize=(16, 12))
scatter_x = range(dct.ravel().size)
scatter_y = np.log10(np.abs(dct.ravel()))
print(scatter_x)
print(scatter_y.shape)
print(scatter_y)
plt.scatter(scatter_x, scatter_y, c='#348ABD', alpha=.3)

'''
 The point is instead of comparing to only Root Mean Squared Error(RMMS) to learn where to stop in the coefficients,
 one could check better metrics which consider visual fidelity or even perceived quality to find 
 the sweet spot between compression ratio and image quality.
 It is easy to get very large coefficients and reject very small coefficients in the reconstructed image 
 but not very easy to either include or reject middle values based on their solely amplitudes. 
 In these coefficients, one needs to look at the frequencies that they belong to, 
 if they are in somehow high frequency range, then it would be rejected 
 whereas if they belong to lower frequency range, it may introduce noticeable and large artifacts into the signal.
 ----
 so basically small coefficients means that the basis image corresponding to that coefficient does not play
 a large part in defining the original image itself thus those can be neglected.
 but if the values are all middlish with all the basis images contributing about the same to the original image
 then we need to turn our eyes to the frequencies of the coefficients which basically means we check out whether
 they are talking about the the slow changes and constant colors (low frequency information) 
 or the edges and fast transition information (high frequency information).
 this can be clearly seen as u,v increases the basis images start to have greater gradients.
'''
plt.title('DCT Coefficient Amplitude vs. Order of Coefficient')
plt.xlabel('Order of DCT Coefficients')
plt.ylabel('DCT Coefficients Amplitude in log scale')

'''
If we look at the first first 2500 coefficients in a 50x50 grid, then we could see that a lot of the coefficients are actually very small 
comparing to the few very large ones. This not only provides a good compaction for the image(less coefficients means high compaction rate), 
but also provides a good compromise between compression and image quality.
{Generally, very low frequencies have a higher ratio of magnitude
orders and similar to very high frequencies.} ???!!
* less coefficients means you get to convey a ok approximation of the image with far less space.
'''
print(np.abs(dct[:10, :10]))
plt.matshow(np.abs(dct[:10, :10]), cmap=plt.cm.Paired)
#plt.xlim(-0.5, 10.5)
#plt.ylim(-0.5, 10.5)
plt.title('DCT output (10x10) for Class 1')
'''
see This explains what this guy says here https://youtu.be/_bltj_7Ne2c?t=869
see the coefficients in the code is the values in the Transform matrix (after we operate dct on the image)
Now each coefficient holds info on the amount of similarity between the image and the corresponding basis image
(corresponding as in the coefficient itself is a function of u,v as is the basis functions)
now with the FIRST 50 coefficients itself the image is reconstructed to a fair (awesome degree)
'''
fig = plt.figure(figsize=(16, 16))
for ii in range(64):
    plt.subplot(8, 8, ii + 1)
    plt.imshow(reconstructed_images[ii], cmap=plt.cm.gray)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

plt.show()