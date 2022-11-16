

# PIPELINE_CONFIG_PATH="/home/object/models/research/object_detection/training/faster_rcnn_inception_resnet_v2/v2/pipeline.config"
# MODEL_DIR="/home/object/caterina/models/research/object_detection/training/faster_rcnn_inception_resnet_v2/v2/model_outputs"
# CHECKPOINT_DIR="/home/object/caterina/models/research/object_detection/training/faster_rcnn_inception_resnet_v2/v2/model_outputs"

# PIPELINE_CONFIG_PATH="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/mines/centernet/pipeline.config"

# MODEL_DIR="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/mines/centernet_2/model_outputs/"
# CHECKPOINT_DIR="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/mines/centernet_2/model_outputs"


# PATH="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/no_mines/post_cartagena/FASTER-RCNN/data_augmentation/"


# PIPELINE_CONFIG_PATH="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/halimeda/halimeda_new_data/test2/pipeline.config"

# MODEL_DIR="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/halimeda/halimeda_new_data/test2/model_outputs"
PIPELINE_CONFIG_PATH="/home/plome/models/research/object_detection/entrenos/pipelines/faster_v1.config"

MODEL_DIR="/home/plome/models/research/object_detection/entrenos/halimeda/faster_v1/model_outputs"
CHECKPOINT_DIR="/home/plome/models/research/object_detection/entrenos/halimeda/faster_v1/model_outputs"

NUM_TRAIN_STEPS=40000
POST_TRAIN_EVAL=False

# sudo sync
# sudo sysctl -w vm.drop_caches=3
# sudo sync
cd /home/plome/models/research/object_detection/

python3 model_main_tf2.py \
    --pipeline_config_path=$PIPELINE_CONFIG_PATH \
    --model_dir=$MODEL_DIR \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --post_train_evaluation=$POST_TRAIN_EVAL \
    --num_train_steps=$NUM_TRAIN_STEPS \
    --alsologtostderr


