"""
Inference script modified from 
https://app.neptune.ai/anton-morgunov/tf-test/n/model-for-inference-36c9b0c4-8d20-4d5a-aa54-5240cc8ce764/6f67c0e3-283c-45de-ae56-405aecd736c0

"""


from importlib.resources import path
import os # importing OS in order to make GPU visible
import tensorflow as tf # import tensorflow

# other import
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import shutil 
from tqdm import tqdm
import sys # importyng sys in order to access scripts located in a different folder¡
import glob
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import imageio
path2scripts = '/home/object/caterina/tf_OD_API/models/research' # TODO: provide pass to the research folder
sys.path.insert(0, path2scripts) # making scripts in models/research available for import
# importing all scripts that will be needed to export your model and use it for inference
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

import scipy
from natsort import natsorted

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # do not change anything in here

# specify which device you want to work on.
# Use "-1" to work on a CPU. Default value "0" stands for the 1st GPU that will be used
os.environ["CUDA_VISIBLE_DEVICES"]="0" # TODO: specify your computational device

# checking that GPU is found
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


#PATHS CATERINA
PATH_TO_CKPT="/home/object/caterina/tf_OD_API/models/research/object_detection/exported_models/mines/test3"
PATH_TO_TEST_IMAGES="/home/object/caterina/tf_OD_API/models/research/object_detection/test_images/mines_online"
PATH_TO_OUTPUT_DIR="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/mines/test3/results_online"
PATH_TO_LABELS="/home/object/caterina/tf_OD_API/models/research/object_detection/data/mines/label_map.pbtx"
PIPELINE_CONFIG_PATH="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/mines/test3/pipeline.config"
MODEL_DIR="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/mines/test3/model_outputs"


# NOTE: your current working directory should be Tensorflow.
# TODO: specify two pathes: to the pipeline.config file and to the folder with trained model.
path2config ='/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/mines/test3/pipeline.config'
path2model = '/home/object/caterina/tf_OD_API/models/research/object_detection/exported_models/mines/test3/checkpoint/'
path2label_map = PATH_TO_LABELS # TODO: provide a path to the label map file

if os.path.exists(PATH_TO_OUTPUT_DIR):
    shutil.rmtree(PATH_TO_OUTPUT_DIR)
if not os.path.exists(PATH_TO_OUTPUT_DIR):
    os.mkdir(PATH_TO_OUTPUT_DIR)

configs = config_util.get_configs_from_pipeline_file(path2config) # importing config
model_config = configs['model'] # recreating model config
detection_model = model_builder.build(model_config=model_config, is_training=False) # importing model

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(path2model, 'ckpt-0')).expect_partial()


def detect_fn(image):
    """
    Detect objects in image.
    
    Args:
      image: (tf.tensor): 4D input image
      
    Returs:
      detections (dict): predictions that model made
    """

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      numpy array with shape (img_height, img_width, 3)
    """
    
    return np.array(Image.open(path))

def inference_as_raw_output(image_path,box_th = 0.25,
                            nms_th = 0.9,
                            to_file = False,
                            data = None,
                            path2dir = False):
    """
    Function that performs inference and return filtered predictions

    Args:
        path2images: an array with pathes to images
        box_th: (float) value that defines threshold for model prediction. Consider 0.25 as a value.
        nms_th: (float) value that defines threshold for non-maximum suppression. Consider 0.5 as a value.
        to_file: (boolean). When passed as True => results are saved into a file. Writing format is
        path2image + (x1abs, y1abs, x2abs, y2abs, score, conf) for box in boxes
        data: (str) name of the dataset you passed in (e.g. test/validation)
        path2dir: (str). Should be passed if path2images has only basenames. If full pathes provided => set False.
        
    Returs:
        detections (dict): filtered predictions that model made
    """

    img_id=os.path.split(image_path)[-1].split(".")[0]
    image_np = load_image_into_numpy_array(image_path)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    # checking how many detections we got
    num_detections = int(detections.pop('num_detections'))
    
    # filtering out detection in order to get only the one that are indeed detections
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    # defining what we need from the resulting detection dict that we got from model output
    key_of_interest = ['detection_classes', 'detection_boxes', 'detection_scores']
    
    # filtering out detection dict in order to get only boxes, classes and scores
    detections = {key: value for key, value in detections.items() if key in key_of_interest}
    
    if box_th: # filtering detection if a confidence threshold for boxes was given as a parameter
        for key in key_of_interest:
            scores = detections['detection_scores']
            current_array = detections[key]
            filtered_current_array = current_array[scores > box_th]
            detections[key] = filtered_current_array
    
    if nms_th: # filtering rectangles if nms threshold was passed in as a parameter
        # creating a zip object that will contain model output info as
        output_info = list(zip(detections['detection_boxes'],
                                detections['detection_scores'],
                                detections['detection_classes']
                                )
                            )
        boxes, scores, classes = nms(output_info)
        
        detections['detection_boxes'] = boxes # format: [y1, x1, y2, x2]
        detections['detection_scores'] = scores
        detections['detection_classes'] = classes

        
    if to_file and data: # if saving to txt file was requested
        print("HELLO THEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
        image_h, image_w, _ = image_np.shape
        file_name = f'pred_result_{img_id}.txt'
        file_name=str(PATH_TO_OUTPUT_DIR)+"/"+file_name
        line2write = list()
        line2write.append(os.path.basename(image_path))
        
        with open(file_name, 'a+') as text_file:
            # iterating over boxes
            for b, s, c in zip(boxes, scores, classes):
                
                y1abs, x1abs = b[0] * image_h, b[1] * image_w
                y2abs, x2abs = b[2] * image_h, b[3] * image_w
                
                # list2append = [x1abs, y1abs, x2abs, y2abs, s, c]
                list2append = [c,s,x1abs, y1abs, x2abs, y2abs]
                line2append = ','.join([str(item) for item in list2append])
                
                line2write.append(line2append)
            
            line2write = ' '.join(line2write)
            text_file.write(line2write + os.linesep)
        
    return detections


def nms(rects, thd=0.5):
    """
    Filter rectangles
    rects is array of oblects ([x1,y1,x2,y2], confidence, class)
    thd - intersection threshold (intersection divides min square of rectange)
    """
    out = []

    remove = [False] * len(rects)

    for i in range(0, len(rects) - 1):
        if remove[i]:
            continue
        inter = [0.0] * len(rects)
        for j in range(i, len(rects)):
            if remove[j]:
                continue
            inter[j] = intersection(rects[i][0], rects[j][0]) / min(square(rects[i][0]), square(rects[j][0]))

        max_prob = 0.0
        max_idx = 0
        for k in range(i, len(rects)):
            if inter[k] >= thd:
                if rects[k][1] > max_prob:
                    max_prob = rects[k][1]
                    max_idx = k

        for k in range(i, len(rects)):
            if (inter[k] >= thd) & (k != max_idx):
                remove[k] = True

    for k in range(0, len(rects)):
        if not remove[k]:
            out.append(rects[k])

    boxes = [box[0] for box in out]
    scores = [score[1] for score in out]
    classes = [cls[2] for cls in out]
    return boxes, scores, classes


def intersection(rect1, rect2):
    """
    Calculates square of intersection of two rectangles
    rect: list with coords of top-right and left-boom corners [x1,y1,x2,y2]
    return: square of intersection
    """
    x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]));
    y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]));
    overlapArea = x_overlap * y_overlap;
    return overlapArea


def square(rect):
    """
    Calculates square of rectangle
    """
    return abs(rect[2] - rect[0]) * abs(rect[3] - rect[1])



def main():
    while(True):
        file_list=os.listdir(PATH_TO_TEST_IMAGES)
        file_list=natsorted(file_list)
        
        if len(file_list) > 0:
            print(file_list)
            file_name=os.path.join(PATH_TO_TEST_IMAGES,file_list[0])
            print(file_name)
            predictions=inference_as_raw_output(file_name,box_th = 0.0, nms_th = 1, to_file = True, data = "Test4",path2dir = False)
            os.remove(file_name)


    print("PREDICTIONS:",predictions)

main()