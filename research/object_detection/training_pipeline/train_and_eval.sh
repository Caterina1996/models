

PIPELINE_CONFIG_PATH="/home/object/caterina/models/research/object_detection/training/faster_rcnn_inception_resnet_v2/v3/pipeline.config"
MODEL_DIR="/home/object/caterina/models/research/object_detection/training/faster_rcnn_inception_resnet_v2/v3/model_outputs"
# CHECKPOINT_DIR="/home/object/caterina/models/research/object_detection/training/faster_rcnn_inception_resnet_v2/v3/model_outputs"
NUM_TRAIN_STEPS=5000
CHECKPOINT_EVERY_N=100
SAMPLE_1_OF_N_EVAL_EXAMPLES=1

python3 model_main_tf2.py --model_dir=$MODEL_DIR --num_train_steps=$NUM_TRAIN_STEPS \
  --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
  --pipeline_config_path=$PIPELINE_CONFIG_PATH \
  # --checkpoint_dir=$CHECKPOINT_DIR \
  --checkpoint_every_n=$CHECKPOINT_EVERY_N \
  --alsologtostderr