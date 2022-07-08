
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # -1 --> Do not use CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
conda_path="/home/sparus/anaconda3/bin:$PATH"
import sys

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


path2scripts = '/home/sparus/object_detection/models/research' # TODO: provide pass to the research folder
sys.path.insert(0, path2scripts) # making scripts in models/research available for import
# importing all scripts that will be needed to export your model and use it for inference
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
from object_detection.builders import model_builder

# sys.path.remove(path2scripts)
# print(sys.path)

# #afegir ros al puthon path per a que robi les llibreries de ros
# # sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

# if conda_path in sys.path:
# 	print(sys.path)
# 	sys.path.remove(conda_path)

# if path2scripts in sys.path:
# 	print(sys.path)
# 	sys.path.remove(path2scripts)

# print(sys.path)

import rospy
import message_filters
import sensor_msgs.msg
from sensor_msgs.msg import Image, CameraInfo

#from cv_bridge import CvBridge, CvBridgeError

class Object_detection:


	def __init__(self, name):
		self.name = name

		# Params
		self.box_th = 0.7
		self.nms_th = 0.5
		self.period = 2
		self.init = False
		self.new_image = False
		PATH="/home/sparus/object_detection/models/research/object_detection/exported_models/no_mines_1000"
		self.path2config = PATH + '/pipeline.config'
		self.path2model = PATH + '/checkpoint/'
		self.path2labels = PATH + '/label_map.pbtx'

		self.category_index = label_map_util.create_category_index_from_labelmap(self.path2labels,use_display_name=True)


		# TODO get paths from rospy.get_param e.g.
		#self.period = rospy.get_param('/object_detection/period')
		#self.path2model = rospy.get_param('/object_detection/model_path')
		#self.path2config=rospy.get_param('')
		#self.path2labels=rospy.get_param('')

		# Set subscribers
		image_sub = message_filters.Subscriber('/stereo_down/scaled_x2/left/image_rect_color', Image)
		info_sub = message_filters.Subscriber('/stereo_down/left/camera_info', CameraInfo)
		image_sub.registerCallback(self.cb_image)
		info_sub.registerCallback(self.cb_info)
		#ts_image = message_filters.TimeSynchronizer([image_sub, info_sub], 10)
		#ts_image.registerCallback(self.cb_data)

		# Set publishers
		# TODO self.pub_mine_det = rospy.Publisher('mine_det', mine_detection, queue_size=4)
		self.pub_mine_bb = rospy.Publisher('image_mine_bb', Image, queue_size=4)

		# Set classification timer
		rospy.Timer(rospy.Duration(self.period), self.run)

		# CvBridge for image conversion
		#self.bridge = CvBridge()


	def cb_image(self, image):
		self.image = image
		self.new_image = True



	def cb_info(self, info):
		self.info = info
		self.new_info = True


	# def cb_data(self, image, info):     # TODO CHECK
	# 	self.image = image
	# 	self.info = info
	# 	self.new_data = True
	

	def set_model(self):
		configs = config_util.get_configs_from_pipeline_file(self.path2config) # importing config
		model_config = configs['model'] # recreating model config
		self.detection_model = model_builder.build(model_config=model_config, is_training=False) # importing model

		ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
		ckpt.restore(os.path.join(self.path2model, 'ckpt-0')).expect_partial()


	def run(self,_):

		# New image available
		if not self.new_image:
			return
		self.new_image = False
		print("New image")

		try:
			image = self.image
			header = self.image.header
			if not self.init:
				rospy.loginfo('[%s]: Start object detection', self.name)	
		except:
			rospy.logwarn('[%s]: There is no input image to run the detection', self.name)
			return

		# Set model
		if not self.init: 
			self.set_model()
			self.init = True
			print("Model init")

		# Object detection
		#image_cv = self.bridge.imgmsg_to_cv2(image, "rgb8")   # TODO CHECK
		#image_np = np.asarray(image_cv)					  # TODO CHECK
		image_np = np.array(np.frombuffer(image.data, dtype=np.uint8).reshape(720, 960,3))
		input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
		detections = self.detection(input_tensor)

		# check number of detections
		num_detections = int(detections.pop('num_detections'))
		# filter out detections
		detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
		# detection_classes to ints
		detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
		# defining what we need from the resulting detection dict that we got from model output
		key_of_interest = ['detection_classes', 'detection_boxes', 'detection_scores']
		# filter out detections dict to get only boxes, classes and scores
		detections = {key: value for key, value in detections.items() if key in key_of_interest}
		if self.box_th != 0: # filtering detection if a confidence threshold for boxes was given as a parameter
			for key in key_of_interest:
				scores = detections['detection_scores']
				current_array = detections[key]
				filtered_current_array = current_array[scores > self.box_th]
				detections[key] = filtered_current_array
		
		if self.nms_th != 0: # filtering rectangles if nms threshold was passed in as a parameter
			# creating a zip object that will contain model output info as
			output_info = list(zip(detections['detection_boxes'],
									detections['detection_scores'],
									detections['detection_classes']))
			boxes, scores, classes = self.nms(output_info, self.nms_th)
			
			detections['detection_boxes'] = boxes # format: [y1, x1, y2, x2]
			detections['detection_scores'] = scores
			detections['detection_classes'] = classes
			
			print(scores)
			
		#print("detections",detections)

		# get used image with printed BBs
		#label_id_offset = 1
		#image_np_bb = image_np.copy()
		#viz_utils.visualize_boxes_and_labels_on_image_array(
		#		image_np_bb,
		#		detections['detection_boxes'],
		#		detections['detection_classes']+label_id_offset,
		#		detections['detection_scores'],
		#		self.category_index,
		#		use_normalized_coordinates=True,
		#		max_boxes_to_draw=200,
		#		min_score_thresh=self.box_th,
		#		agnostic_mode=False,
		#		line_thickness=5)

		# Publishers
		# TODO publish image and info togheter

		#image_bb = self.bridge.cv2_to_imgmsg(image_cv_bb, encoding="bgr8")  # TODO CHECK
		#image_bb = self.msgify(Image, image_np_bb, encoding='rgb8')
		#image_bb.header = header
		#self.pub_mine_bb.publish(image_bb)


	#def msgify(msg_type, numpy_obj, *args, **kwargs):
	#	conv = _from_numpy.get((msg_type, kwargs.pop('plural', False)))
	#	return conv(numpy_obj, *args, **kwargs)


	def detection(self,image):
		"""
		Detect objects in image.

		Args:
		image: (tf.tensor): 4D input image

		Returs:
		detections (dict): predictions that model made
		"""

		image, shapes = self.detection_model.preprocess(image)
		prediction_dict = self.detection_model.predict(image, shapes)
		detections = self.detection_model.postprocess(prediction_dict, shapes)
		return detections


	def nms(self,rects, thd=0.5):
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
				inter[j] = self.intersection(rects[i][0], rects[j][0]) / min(self.square(rects[i][0]), self.square(rects[j][0]))

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


	def intersection(self,rect1, rect2):
		"""
		Calculates square of intersection of two rectangles
		rect: list with coords of top-right and left-boom corners [x1,y1,x2,y2]
		return: square of intersection
		"""
		x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]));
		y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]));
		overlapArea = x_overlap * y_overlap;
		return overlapArea


	def square(self,rect):
		"""
		Calculates square of rectangle
		"""
		return abs(rect[2] - rect[0]) * abs(rect[3] - rect[1])


if __name__ == '__main__':
	try:
		rospy.init_node('segment_image')
		Object_detection(rospy.get_name())

		rospy.spin()
	except rospy.ROSInterruptException:
		pass
