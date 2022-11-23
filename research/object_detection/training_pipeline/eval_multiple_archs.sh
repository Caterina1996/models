
PIPELINE_CONFIG_PATH="/home/plome/models/research/object_detection/entrenos/pipelines/efficient_d3_lr1.config"
MODEL_DIR="/home/plome/models/research/object_detection/entrenos/halimeda/efficient_d3/lr1/model_outputs"
CHECKPOINT_DIR="/home/plome/models/research/object_detection/entrenos/halimeda/efficient_d3/lr1/model_outputs"

NUM_TRAIN_STEPS=100000
POST_TRAIN_EVAL=False


cd /home/plome/models/research/object_detection/

python3 model_main_tf2.py \
    --pipeline_config_path=$PIPELINE_CONFIG_PATH \
    --model_dir=$MODEL_DIR \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --post_train_evaluation=$POST_TRAIN_EVAL \
    --num_train_steps=$NUM_TRAIN_STEPS \
    --alsologtostderr

PIPELINE_CONFIG_PATH="/home/plome/models/research/object_detection/entrenos/pipelines/efficient_d3_lr2.config"
MODEL_DIR="/home/plome/models/research/object_detection/entrenos/halimeda/efficient_d3/lr2/model_outputs"
CHECKPOINT_DIR="/home/plome/models/research/object_detection/entrenos/halimeda/efficient_d3/lr2/model_outputs"

cd /home/plome/models/research/object_detection/

python3 model_main_tf2.py \
    --pipeline_config_path=$PIPELINE_CONFIG_PATH \
    --model_dir=$MODEL_DIR \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --post_train_evaluation=$POST_TRAIN_EVAL \
    --num_train_steps=$NUM_TRAIN_STEPS \
    --alsologtostderr

PIPELINE_CONFIG_PATH="/home/plome/models/research/object_detection/entrenos/pipelines/efficient_d3_lr2.config"
MODEL_DIR="/home/plome/models/research/object_detection/entrenos/halimeda/efficient_d3/lr2/model_outputs"
CHECKPOINT_DIR="/home/plome/models/research/object_detection/entrenos/halimeda/efficient_d3/lr2/model_outputs"

cd /home/plome/models/research/object_detection/

python3 model_main_tf2.py \
    --pipeline_config_path=$PIPELINE_CONFIG_PATH \
    --model_dir=$MODEL_DIR \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --post_train_evaluation=$POST_TRAIN_EVAL \
    --num_train_steps=$NUM_TRAIN_STEPS \
    --alsologtostderr