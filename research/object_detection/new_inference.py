"""
Get inference from saved model

Script created following Tensorflow Object Detection API:
https://github.com/Caterina1996/models/blob/master/research/object_detection/colab_tutorials/inference_tf2_colab.ipynb


Check this other options too:
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/plot_object_detection_checkpoint.html


https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/inference_tf2_colab.ipynb

"""


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

# import tensorflow as tf
import tensorflow.compat.v1 as tf
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import dataset_util, label_map_util
# matplotlib inline


flags = tf.app.flags
flags.DEFINE_string('model_checkpoint_dir', '', 'Directory where the saved trained model is stored.')
flags.DEFINE_string('print_thr', '', 'Printing confidence threshold.')
flags.DEFINE_string('test_dir', '', 'test images folder path.')
flags.DEFINE_string('out_dir', 'results', 'output folder path.')
flags.DEFINE_string('id', '', 'identifier.')
flags.DEFINE_string('n_classes', '', 'number of classes.')
flags.DEFINE_string('labels_file', '', 'path to the class label file')

FLAGS = flags.FLAGS
PATH_TO_LABELS = FLAGS.labels_file
PATH_TO_CKPT = FLAGS.model_checkpoint_dir 

PATH_TO_TEST_IMAGES = FLAGS.test_dir
PATH_TO_OUTPUT_DIR = FLAGS.out_dir


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: a file path (this can be local or on colossus)

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """

    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

#LOAD LABELS

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


imgs_paths= os.listdir(PATH_TO_TEST_IMAGES)

for image in imgs_paths:

    if re.search("\.(png|jpg|jpeg|JPG|JPEG)$", image):
        image_dir=PATH_TO_TEST_IMAGES+"/"+image

        image_np = load_image_into_numpy_array(image_path)
        input_tensor = np.expand_dims(image_np, 0)
        # input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32) according to tf tutorial documentation 

        start_time = time.time()
        detections = detect_fn(input_tensor)
        end_time = time.time()
        elapsed.append(end_time - start_time)

        plt.rcParams['figure.figsize'] = [42, 21]
        label_id_offset = 1
        image_np_with_detections = image_np.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'][0].numpy(),
                detections['detection_classes'][0].numpy().astype(np.int32),
                detections['detection_scores'][0].numpy(),
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.40,
                agnostic_mode=False)
        plt.subplot(2, 1, i+1)
        plt.imshow(image_np_with_detections)


    else:
        #informative warning! wrong file type!
        print("WRONG FILE TYPE -------------NOT AN IMAGE !!!! CURRENT FILE NAME IS ",image)

mean_elapsed = sum(elapsed) / float(len(elapsed))
print('Elapsed time: ' + str(mean_elapsed) + ' second per image')