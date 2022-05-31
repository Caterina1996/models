import os
import pathlib
import io
import os
import scipy.misc
import numpy as np
import six
import time

from six import BytesIO

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
from utils import label_map_util
import re


import io
import scipy.misc

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

#EXAMPLE WORKING:

# PATH_TO_CKPT="exported_models/faster_rcnn_inception_resnet_v2/v2/exported_model"
# PATH_TO_TEST_IMAGES="test_images/halimeda_test"
# PATH_TO_OUTPUT_DIR="training/faster_rcnn_inception_resnet_v2/v2/results/"
# PATH_TO_LABELS="/home/object/models/research/object_detection/data/halimeda_test/label_map.pbtxt"



PATH_TO_CKPT="/home/object/caterina/tf_OD_API/models/research/object_detection/exported_models/halimeda_nodataug"

PATH_TO_TEST_IMAGES="/home/object/caterina/tf_OD_API/models/research/object_detection/test_images/halimeda_new_test"

PATH_TO_OUTPUT_DIR="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/new_halimeda_test/test_nodataug/results"

# OUTPUT_DIRECTORY_PATH="/home/object/caterina/tf_OD_API/models/research/object_detection/results/new_halimeda_test"
PATH_TO_LABELS="/home/object/caterina/tf_OD_API/models/research/object_detection/data/halimeda_new_test/label_map.pbtxt"

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


def plot_inference(image_path,image):
    image_np = load_image_into_numpy_array(image_path)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    fig=plt.figure(frameon=False)    
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(image_np_with_detections, aspect='auto')
    plt.imshow(image_np_with_detections)
    # plt.savefig(PATH_TO_OUTPUT_DIR+image,bbox_inches='tight', pad_inches=0)
    fig.savefig(PATH_TO_OUTPUT_DIR+image, bbox_inches='tight', pad_inches=0)
    print('Image saved in '+PATH_TO_OUTPUT_DIR+image)
    plt.show()

#---------------------------------------------------------------------------------------

imgs_paths= os.listdir(PATH_TO_TEST_IMAGES)

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,max_num_classes=label_map_util.get_max_label_map_index(label_map),use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

start_time = time.time()
tf.keras.backend.clear_session()
detect_fn = tf.saved_model.load(PATH_TO_CKPT+'/saved_model/')
end_time = time.time()
elapsed_time = end_time - start_time
print('Elapsed time: ' + str(elapsed_time) + 's')
elapsed=[]

for image in imgs_paths:

    if re.search("\.(png|jpg|jpeg|JPG|JPEG)$", image):
        image_dir=PATH_TO_TEST_IMAGES+"/"+image
        plot_inference(image_dir,image)

    else:
        #informative warning! wrong file type!
        print("WRONG FILE TYPE -------------NOT AN IMAGE !!!! CURRENT FILE NAME IS ",image)

