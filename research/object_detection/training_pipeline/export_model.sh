

#EXAMPLE OF USAGE TUTORIAL
# python exporter_main_v2.py \
#   --pipeline_config_path=<path to a config file> \
#   --trained_checkpoint_dir=<path to a directory with your trained model> \
#   --output_directory=<path to a directory where to export a model> \
#   --input_type=image_tensor

# pipeline_config_path -> <path to your config file> 
# path to the config file for the model you want to export. Should be a config file from ./models/<folder with the model of your choice>/v1/ 

# trained_checkpoint_dir -> <path to a directory with your trained model> 
# path to a directory where model checkpoints were placed during training. Should also be the following: ./models/<folder with the model of your choice>/v1/  

# output_directory -> <path to a directory where to export a model> 
# path where an exported model will be saved. Should be: ./exported_models/<folder with the model of your choice> 


# Example Usage Docs:
# --------------
# python exporter_main_v2.py \
#     --input_type image_tensor \
#     --pipeline_config_path path/to/ssd_inception_v2.config \
#     --trained_checkpoint_dir path/to/checkpoint \
#     --output_directory path/to/exported_model_directory
#     --use_side_inputs True/False \
#     --side_input_shapes dim_0,dim_1,...dim_a/.../dim_0,dim_1,...,dim_z \
#     --side_input_names name_a,name_b,...,name_c \
#     --side_input_types type_1,type_2


# MODEL_CHECKPOINT_DIR="/home/object/models/research/object_detection/training/faster_rcnn_inception_resnet_v2/v1/pipeline.config"
# TRAINDED_CHECKPOINT_DIR="/home/object/models/research/object_detection/training/faster_rcnn_inception_resnet_v2/v1/model_outputs"
# OUTPUT_DIRECTORY_PATH="/home/object/models/research/object_detection/exported_models/faster_rcnn_inception_resnet_v2/"

#MINES
# PIPELINE_CONFIG_PATH="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/mines/FASTR-CNN/dataug/mines_frames_bckgrnd/pipeline.config"
# TRAINDED_CHECKPOINT_DIR="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/mines/FASTR-CNN/dataug/mines_frames_bckgrnd/to_freeze"
# OUTPUT_DIRECTORY_PATH="/home/object/caterina/tf_OD_API/models/research/object_detection/exported_models/mines/FASTR-CNN/dataug/mines_frames_bckgrnd/frozen_20k/"


PIPELINE_CONFIG_PATH="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/halimeda/halimeda_new_data/FASTER-RCNN/selec_03_10_22/pipeline.config"
# PIPELINE_CONFIG_PATH="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/halimeda/halimeda_new_data/SSD_RESNET50v1_1024/test1/pipeline.config"

# TRAINDED_CHECKPOINT_DIR="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/halimeda/halimeda_new_data/SSD_RESNET50v1_1024/test1/model_outputs"
TRAINDED_CHECKPOINT_DIR="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/halimeda/halimeda_new_data/FASTER-RCNN/selec_03_10_22/model_outputs"

OUTPUT_DIRECTORY_PATH="/home/object/caterina/tf_OD_API/models/research/object_detection/exported_models/halimeda/FASTER-RCNN/selec_03_10_22/frozen_40k"
# OUTPUT_DIRECTORY_PATH="/home/object/caterina/tf_OD_API/models/research/object_detection/exported_models/halimeda/SSD_RESNET50v1_1024/test1/frozen_20k"


cd /home/object/caterina/tf_OD_API/models/research/object_detection/

python3 exporter_main_v2.py \
  --pipeline_config_path=$PIPELINE_CONFIG_PATH \
  --trained_checkpoint_dir=$TRAINDED_CHECKPOINT_DIR \
  --output_directory=$OUTPUT_DIRECTORY_PATH \
  --input_type=image_tensor

#Console output: 
#Writing pipeline config file to /home/object/models/research/object_detection/exported_models/faster_rcnn_inception_resnet_v2/pipeline.config
