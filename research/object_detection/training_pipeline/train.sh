# Params according to https://neptune.ai/blog/how-to-train-your-own-object-detector-using-tensorflow-object-detection-api
# path to your config file> 
# is a path to the config file you are going to use for the current training job. Should be a config file from ./models/<folder with the model of your choice>/v1/ 
# <path to a directory with your model> 
# is a path to a directory where all of your future model attributes will be placed. Should also be the following: ./models/<folder with the model of your choice>/v1/  
# <int for the number of steps per checkpoint> 
# is an integer that defines how many steps should be completed in a sequence order to make a model checkpoint. Remember, that when a single step is made, your model processes a number of images equal to your batch_size defined for training.


# PIPELINE_CONFIG_PATH="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/mines/centernet/pipeline.config"
# PIPELINE_CONFIG_PATH="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/mines/test3/pipeline.config"
# MODEL_DIR="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/mines/centernet_2/model_outputs"
# PIPELINE_CONFIG_PATH="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/mines/efficientDetD6/1/effiD6_lrcosine_min.config"

# PIPELINE_CONFIG_PATH="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/mines/efficientD5/pipeline.config"

# PATH="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/no_mines/post_cartagena/FASTER-RCNN/data_augmentation/"

#TODO: TEST RESNET 50
#funciona:
# PIPELINE_CONFIG_PATH="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/halimeda/halimeda_new_data/SSD_RESNET50v1_1024/test1/pipeline.config"

PIPELINE_CONFIG_PATH="/home/plome/models/research/object_detection/entrenos/pipelines/faster_v1.config"

MODEL_DIR="/home/plome/models/research/object_detection/entrenos/halimeda/faster_v1/model_outputs"

NUM_TRAIN_STEPS=40000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
CHECKPOINT_EVERY_N=1000  #num steps fets per guardar un checkpoint.Per cada step el model processa batch_size imatges
NUM_STEPS_PER_ITERATION=100
CHECKPOINT_MAX_TO_KEEP=200


# num_workers-> Amb aquest parametre podem indicar quants dels cores de la CPU (que a Olivia és multicore) volem usar.
#Pot ser amb num_workers limitat ja no petaria? -> Posar al màxim?
# cp train.sh /home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/mines/centernet/model_outputs/train.sh

cp train.sh $MODEL_DIR/train.sh
cd /home/plome/models/research/object_detection/

#clean cache memory ! check ram usage free -m
# sudo sync
# sudo sysctl -w vm.drop_caches=3
# sudo sync

    
python3 model_main_tf2.py --model_dir=$MODEL_DIR --num_train_steps=$NUM_TRAIN_STEPS \
  --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
  --pipeline_config_path=$PIPELINE_CONFIG_PATH \
  --checkpoint_every_n=$CHECKPOINT_EVERY_N \
  --num_steps_per_iteration=$NUM_STEPS_PER_ITERATION \
  --checkpoint_max_to_keep=$CHECKPOINT_MAX_TO_KEEP \
  --alsologtostderr


# python3 model_main_tf2.py --model_dir="/home/object/caterina/models/research/object_detection/object_ws/trainings/test_parametres/model_outputs" \
#   --num_train_steps=200 \
#   --sample_1_of_n_eval_examples=1 \
#   --pipeline_config_path="/home/object/caterina/models/research/object_detection/object_ws/trainings/test_parametres/pipeline.config" \
#   --checkpoint_every_n=50 \
#   --alsologtostderr
