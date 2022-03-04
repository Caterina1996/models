


PIPELINE_CONFIG_PATH="/home/object/models/research/object_detection/training/faster_rcnn_inception_resnet_v2/v1/pipeline.config"
MODEL_DIR="/home/object/models/research/object_detection/training/faster_rcnn_inception_resnet_v2/v1/model_outputs"
CHECKPOINT_DIR="/home/object/models/research/object_detection/training/faster_rcnn_inception_resnet_v2/v1/model_outputs"

MODEL_DIR={path to model directory}
python3 model_main_tf2.py \
    --pipeline_config_path=$PIPELINE_CONFIG_PATH \
    --model_dir=$MODEL_DIR \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --alsologtostderr

