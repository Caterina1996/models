# Params according to https://neptune.ai/blog/how-to-train-your-own-object-detector-using-tensorflow-object-detection-api
# path to your config file> 
# is a path to the config file you are going to use for the current training job. Should be a config file from ./models/<folder with the model of your choice>/v1/ 
# <path to a directory with your model> 
# is a path to a directory where all of your future model attributes will be placed. Should also be the following: ./models/<folder with the model of your choice>/v1/  
# <int for the number of steps per checkpoint> 
# is an integer that defines how many steps should be completed in a sequence order to make a model checkpoint. Remember, that when a single step is made, your model processes a number of images equal to your batch_size defined for training.


PIPELINE_CONFIG_PATH="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/new_halimeda_test/pipeline.config"

MODEL_DIR="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/new_halimeda_test/model_OUT_SMALL_DS"

NUM_TRAIN_STEPS=10000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
CHECKPOINT_EVERY_N=100  #num steps fets per guardar un checkpoint.Per cada step el model processa batch_size imatges
NUM_STEPS_PER_ITERATION=100
CHECKPOINT_MAX_TO_KEEP=800


# num_workers-> Amb aquest parametre podem indicar quants dels cores de la CPU (que a Olivia és multicore) volem usar.
#Pot ser amb num_workers limitat ja no petaria? -> Posar al màxim?

cd /home/object/caterina/tf_OD_API/models/research/object_detection/

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
