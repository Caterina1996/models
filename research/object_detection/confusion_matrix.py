

"""
Inference script modified from 
https://app.neptune.ai/anton-morgunov/tf-test/n/model-for-inference-36c9b0c4-8d20-4d5a-aa54-5240cc8ce764/6f67c0e3-283c-45de-ae56-405aecd736c0

"""


import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # -1 --> Do not use CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
conda_path="/home/sparus/anaconda3/bin:$PATH"
import sys
import pandas as pd
# print(sys.path)
# #Per importar les llibreries des de l'entorn de conda i no ros
# if ros_path in sys.path:
# 	print(sys.path)
# 	sys.path.remove(ros_path)

import numpy as np
import tensorflow as tf
import scipy
import shutil 
# import cv2
import PIL.Image as Image
import glob
path2scripts = '/home/sparus/object_detection/models/research' # TODO: provide pass to the research folder
sys.path.insert(0, path2scripts) # making scripts in models/research available for import
# importing all scripts that will be needed to export your model and use it for inference
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
from object_detection.builders import model_builder

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # do not change anything in here

# specify which device you want to work on.
# Use "-1" to work on a CPU. Default value "0" stands for the 1st GPU that will be used
os.environ["CUDA_VISIBLE_DEVICES"]="0" # TODO: specify your computational device

# checking that GPU is found
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


#EXPORTED MODEL PATH

common2 = "/home/sparus/object_detection/models/research/object_detection/exported_models/mines_frames_bckgrnd/"
path2config =  common2 + 'pipeline.config'
path2model =  common2 + 'checkpoint/'

PATH_TO_LABELS= common2 + 'label_map.pbtx'

PATH_TO_OUTPUT_DIR= "/home/sparus/inference_mines/out/"

if not os.path.exists(PATH_TO_OUTPUT_DIR):
    os.mkdir(PATH_TO_OUTPUT_DIR)
    print("hello!")

# do not change anything in this cell
configs = config_util.get_configs_from_pipeline_file(path2config) # importing config
model_config = configs['model'] # recreating model config
detection_model = model_builder.build(model_config=model_config, is_training=False) # importing model

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(path2model, 'ckpt-0')).expect_partial()

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,use_display_name=True)

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
label_map_dict = label_map_util.get_label_map_dict(label_map)

print("labelmap: ",label_map_dict)


print("labelmap: ",label_map_dict)

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


def plot_bbox_from_csv(path_to_csv, box_th=0.1,plot_gt=False,path_to_gt_csv=None):
    """
    Function reads a df with files and predictions or gts and infers 
    
    Args:
      path_to_csv: path to csv file containing predictions or gts
      Row names are supossed to be [filename	width	height	class	xmin	ymin	xmax	ymax	score]
      output_path: path where images with bbox will be stored
      box_th: (float) value that defines threshold for model prediction.
      is_gt: must be True for groundtruth, False for predictions
      
    Returns:
      None
    """

    df = pd.read_csv(path_to_csv)
    print(df.head())

    image_ids=df["filename"]
    len_df=len(image_ids)
    print("len DATASET",len_df)
    print("repeated ids:",image_ids)
    image_ids=list(set(image_ids))
    print("suposadament aquest es un set",image_ids)

    boxes_coordinate_keys=["ymin", "xmin", "ymax", "xmax"]
    print("COORDINATES SORTED CORRECTLY",df[boxes_coordinate_keys].head())
    
    
    if plot_gt==True:
        df_gt=pd.read_csv(path_to_gt_csv)
        print(df_gt.describe())
        scores=len(df_gt["filename"])*[1]
        df_gt["score"]=scores
        df_gt=df_gt.replace({"class": label_map_dict})
        # label_id_offset=0
    
    label_id_offset = 1
    
    correction=[value +label_id_offset for value in df["class"].values]
    df["class"]=correction
    print(df["class"])

    i=0
    for id in image_ids:
        df_image=df.loc[df['filename'] == id]
        df_image_gt=df_gt.loc[df_gt['filename'] == id]
        print("THIS IS IMAGE DF!!!!")
        print(df_image.head())
        bboxes=df_image[boxes_coordinate_keys].to_numpy()
        bboxes_gt=df_image_gt[boxes_coordinate_keys].to_numpy()
        
        #check this!!!
        detection_classes= df_image["class"].to_numpy()#+label_id_offset
        
        detection_classes_gt= df_image_gt["class"].values.tolist()#+label_id_offset

        detection_scores= df_image["score"].values.tolist()
        detection_scores_gt= df_image_gt["score"].values.tolist()
        
        print("-------------------------------------------------------------")
        print("CLASS OPERATION",df_image["class"])

        print("Bounding Boxes!!!!",bboxes)
        print("-------------------------------------------------------------")

        print("DETECTION CLASSES",detection_classes)
        print("-------------------------------------------------------------")
        print("-------------------------------------------------------------")
        i+=1
        
        print('Running inference for {}... '.format(PATH_TO_TEST_IMAGES), end='')
        image_np = load_image_into_numpy_array(PATH_TO_TEST_IMAGES+"/"+id)
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                bboxes,
                detection_classes,
                detection_scores,
                category_index,
                use_normalized_coordinates=False,
                max_boxes_to_draw=1000,
                min_score_thresh=box_th,
                agnostic_mode=False,
                line_thickness=5)

        if plot_gt:
            #agnostic mode perquè pinti en taronja (total només hi ha una classe)
            viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    bboxes_gt,
                    detection_classes_gt,
                    detection_scores_gt,
                    category_index,
                    use_normalized_coordinates=False,
                    max_boxes_to_draw=1000,
                    min_score_thresh=box_th,
                    agnostic_mode=True,
                    line_thickness=5)
        
        

        if (image_np==image_np_with_detections).all():
            print("TELL ME WHYYYYYYYYYYYYYYYYYYYYYY???????????????")

        img_path=str((PATH_TO_OUTPUT_DIR)+"/"+id)
        image_pil = Image.fromarray(image_np_with_detections)
        imageio.imwrite(img_path, image_pil)

        print('Image saved in '+img_path)
        print('Done')


    return

    
def get_detections(path2images,
                    box_th = 0.10,
                    nms_th = 0.9,
                    data = None):
    """
    Function that performs inference and return filtered predictions

    Args:
        path2images: an array with pathes to images
        box_th: (float) value that defines threshold for model prediction. Consider 0.25 as a value.
        nms_th: (float) value that defines threshold for non-maximum suppression. Consider 0.5 as a value.
        path2image + (x1abs, y1abs, x2abs, y2abs, score, conf) for box in boxes
        data: (str) name of the dataset you passed in (e.g. test/validation)
            
    Returs:
        predictions_df (df): filtered predictions df that model made
    """
    
    print (f'Current data set is {data}')
    print (f'Ready to start inference on {len(path2images)} images!')
    preds_list=[]
    for image_path in path2images:

        print("IMAGE PATH IS:,",image_path,"\n")
        filename=image_path.split("/")[-1]
            
        image_np = load_image_into_numpy_array(image_path)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        print("DETECTIONS ARE: ", detections)
        # checking how many dete
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
        
        column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax','score']
        image_h, image_w, _ = image_np.shape

        for b, s, c in zip(boxes, scores, classes):
                    
                    y1abs, x1abs = b[0] * image_h, b[1] * image_w
                    y2abs, x2abs = b[2] * image_h, b[3] * image_w
                    
                    # prediction = [x1abs, y1abs, x2abs, y2abs, s, c]
                    prediction = [filename ,image_w, image_h, c, x1abs, y1abs, x2abs, y2abs, s]
                    # prediction = [filenam/home/sparus/object_detection/models/research/object_detection/exported_models/no_mines_1000/
        predictions_df = pd.DataFrame(preds_list, columns=column_name)
                 
    return predictions_df



path2images=glob.glob("/home/sparus/inference_mines/imgs/**")

print("Path to images is",path2images)

PATH_TO_OUTPUT_DIR= "/home/sparus/inference_mines/out/"

csvs_path="/home/sparus/inference_mines/out/"

path_to_csv=csvs_path+"predictions.csv"

# path_to_gt_csv=csvs_path+"train_labels.csv"

predict=True

if predict:
    # Guardar les prediccions a un df "readable" per plotejar 
    predictions_df=get_detections(path2images,box_th = 0.10,nms_th = 0.9, data = None)

    predictions_df.to_csv(PATH_TO_OUTPUT_DIR + "/predictions_mines.csv", index=None)

    print("PREDICTIONS:",predictions_df.head())

else:
# plot_bbox(path_to_csv, box_th=0.25,is_gt=True)
    plot_bbox_from_csv(path_to_csv, box_th=0.1,plot_gt=True,path_to_gt_csv=path_to_gt_csv)

# plot_bbox(path2images, box_th=0.25)


