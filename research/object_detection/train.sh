

PIPELINE_CONFIG_PATH="/home/object/models/research/object_detection/training/faster_rcnn_inception_resnet_v2/v1/pipeline.config"
MODEL_DIR="/home/object/models/research/object_detection/training/faster_rcnn_inception_resnet_v2/v1/model_outputs"
NUM_TRAIN_STEPS=50
SAMPLE_1_OF_N_EVAL_EXAMPLES=1

python3 model_main_tf2.py --model_dir=$MODEL_DIR --num_train_steps=$NUM_TRAIN_STEPS \
  --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
  --pipeline_config_path=$PIPELINE_CONFIG_PATH \
  --alsologtostderr