


PIPELINE_CONFIG_PATH="/home/plome/models/research/object_detection/entrenos/pipelines/efficient_d2.config"

MODEL_DIR="/home/plome/models/research/object_detection/entrenos/halimeda/efficient_d2/model_outputs"

CHECKPOINT_DIR="/home/plome/models/research/object_detection/entrenos/halimeda/efficient_d2/model_outputs"

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


