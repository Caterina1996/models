

# PIPELINE_CONFIG_PATH="/home/object/models/research/object_detection/training/faster_rcnn_inception_resnet_v2/v2/pipeline.config"
# MODEL_DIR="/home/object/caterina/models/research/object_detection/training/faster_rcnn_inception_resnet_v2/v2/model_outputs"
# CHECKPOINT_DIR="/home/object/caterina/models/research/object_detection/training/faster_rcnn_inception_resnet_v2/v2/model_outputs"


PIPELINE_CONFIG_PATH="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/new_halimeda_test/pipeline.config"
# path to a directory where the evaluation job will write logs:
MODEL_DIR="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/new_halimeda_test/model_outputs_traineval"


CHECKPOINT_DIR="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/new_halimeda_test/model_outputs_traineval"

# NUM_TRAIN_STEPS=5000
POST_TRAIN_EVAL=False

cd /home/object/caterina/tf_OD_API/models/research/object_detection/

python3 model_main_tf2.py \
    --pipeline_config_path=$PIPELINE_CONFIG_PATH \
    --model_dir=$MODEL_DIR \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --post_train_evaluation=$POST_TRAIN_EVAL \
    --alsologtostderr


