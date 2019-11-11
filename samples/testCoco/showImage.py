import os
# importing io from skimage
import skimage
from skimage import io
# way to load image from file
file = os.path.join(skimage.data_dir, 'C:/Users/tunis/PycharmProjects/Mask_RCNN/samples/test/sample.jpg')
myimg = io.imread(file)
# way to show the input image
io.imshow(myimg)
io.show()