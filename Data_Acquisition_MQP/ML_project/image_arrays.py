import numpy as np
import glob
import imageio
from scipy import ndimage

label = []

for kk in range(0, 18):
     for ii in range(0, 5):
          for jj in range(0, 100):
               label.append(ii)

label = np.asarray(label)
print(label)
print(label.shape)

np.save("P:/MQP_data/ML_project_data/data/subject_3/y_3.npy", label)

image_tensor_mega = []
image_tensor = []

for im_path in glob.glob("P:/MQP_data/ML_project_data/Perpendicular_1/S_1_3/image*.png"):
     print(im_path)
     im = imageio.imread(im_path)
     #cropped size: 400 x 560
     im = im[50:450, 40:600]
     #zoomed in: 100 x 140
     im = ndimage.interpolation.zoom(im, .25)
     #plt.imshow(im, interpolation='nearest')
     #plt.show()
     image_tensor.append(im)

for im_path in glob.glob("P:/MQP_data/ML_project_data/Perpendicular_2/S_2_3/image*.png"):
     print(im_path)
     im = imageio.imread(im_path)
     #cropped size: 400 x 560
     im = im[50:450, 40:600]
     #zoomed in: 100 x 140
     im = ndimage.interpolation.zoom(im, .25)
     #plt.imshow(im, interpolation='nearest')
     #plt.show()
     image_tensor.append(im)

for im_path in glob.glob("P:/MQP_data/ML_project_data/Perpendicular_3/S_3_3/image*.png"):
     print(im_path)
     im = imageio.imread(im_path)
     #cropped size: 400 x 560
     im = im[50:450, 40:600]
     #zoomed in: 100 x 140
     im = ndimage.interpolation.zoom(im, .25)
     #plt.imshow(im, interpolation='nearest')
     #plt.show()
     image_tensor.append(im)

subject_1_perpendicular = np.array(image_tensor)
print(subject_1_perpendicular.shape)

np.save("P:/MQP_data/ML_project_data/data/subject_3/X_3.npy", subject_1_perpendicular)
