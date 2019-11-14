# https://www.analyticsvidhya.com/blog/2019/07/computer-vision-implementing-mask-r-cnn-image-segmentation/
# https://github.com/matterport/Mask_RCNN
# https://github.com/cs-chan/Total-Text-Dataset
import os
import sys
import tkinter
import matplotlib

from samples.totalText import totalText

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import skimage
from skimage import io

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

import warnings

warnings.filterwarnings("ignore")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

sys.path.append(os.path.join(ROOT_DIR, "samples/totalText/"))  # To find local version

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
MODEL_PATH = os.path.join('', "mask_rcnn_totaltext_0002.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(totalText.TotalTextConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(mode="inference", model_dir='mask_rcnn_totaltext_0002.h5', config=config)
model.load_weights('mask_rcnn_totaltext_0002.h5', by_name=True)

class_names = ['BG', 'totalText']

input_directory = 'C:/Users/tunis/Documents/UofM/ECE5831/FinalProject/totaltext/Images/Test/'
output_directory_base = 'C:/Users/tunis/PycharmProjects/Mask_RCNN/samples/totalText/output/'

for filename in os.listdir(input_directory):
    output_directory = output_directory_base + filename.split('.')[0] + '/'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    image = io.imread(os.path.join(skimage.data_dir, input_directory + filename))

    plt.figure(figsize=(12, 10))
    plt.imshow(image)
    plt.savefig(output_directory + 'original.png')
    plt.close()

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    plt.savefig(output_directory + 'output-mask.png')
    plt.close()

    mask = r['masks']
    mask = mask.astype(int)
    print(mask.shape)

    for i in range(mask.shape[2]):
        temp = skimage.io.imread(input_directory + filename)
        for j in range(temp.shape[2]):
            temp[:, :, j] = temp[:, :, j] * mask[:, :, i]
        plt.figure(figsize=(8, 8))
        plt.imshow(temp)
        plt.savefig(output_directory + 'segment' + str(i) + '.png')
        plt.close()






# # Load a random image from the images folder
# sample = 'C:/Users/tunis/Documents/UofM/ECE5831/FinalProject/totaltext/Images/Test/img3.jpg'
# image = io.imread(os.path.join(skimage.data_dir, sample))
#
# # original image
# plt.figure(figsize=(12, 10))
# io.imshow(image)
# plt.savefig('C:/Users/tunis/PycharmProjects/Mask_RCNN/samples/totalText/output/output-orig.png')
#
# # Run detection
# results = model.detect([image], verbose=1)
#
# # Visualize results
# r = results[0]
# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
# plt.savefig('C:/Users/tunis/PycharmProjects/Mask_RCNN/samples/totalText/output/output-mask.png')
#
# mask = r['masks']
# mask = mask.astype(int)
# print(mask.shape)
#
# for i in range(mask.shape[2]):
#     temp = skimage.io.imread(sample)
#     for j in range(temp.shape[2]):
#         temp[:, :, j] = temp[:, :, j] * mask[:, :, i]
#     plt.figure(figsize=(8, 8))
#     plt.imshow(temp)
#     plt.savefig('C:/Users/tunis/PycharmProjects/Mask_RCNN/samples/totalText/output/segment' + str(i) + '.png')
