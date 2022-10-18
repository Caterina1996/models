#training pipeline:

PIPELINE_CONFIG_PATH="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/mines/centernet/pipeline.config"

MODEL_DIR="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/mines/centernet/model_outputs"

CHECKPOINT_DIR="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/mines/centernet/model_outputs"

TRAINDED_CHECKPOINT_DIR="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/no_mines_2k/model_outputs"

OUTPUT_DIRECTORY_PATH="/home/object/caterina/tf_OD_API/models/research/object_detection/exported_models/no_mines_2k"

NUM_TRAIN_STEPS=5000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
CHECKPOINT_EVERY_N=100  #num steps fets per guardar un checkpoint.Per cada step el model processa batch_size imatges
NUM_STEPS_PER_ITERATION=100
CHECKPOINT_MAX_TO_KEEP=200

POST_TRAIN_EVAL=False


#1) Clean caché memory
#check ram usage with free -m

sudo sync
sudo sysctl -w vm.drop_caches=3
sudo sync

# 2) Train:

sh train.sh

gnome-terminal -x sh eval.sh

sh export_model.sh
sh inference.sh


