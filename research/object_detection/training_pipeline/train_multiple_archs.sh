


NUM_TRAIN_STEPS=100000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
CHECKPOINT_EVERY_N=1000  #num steps fets per guardar un checkpoint.Per cada step el model processa batch_size imatges
NUM_STEPS_PER_ITERATION=100
CHECKPOINT_MAX_TO_KEEP=200

PIPELINE_CONFIG_PATH="/home/plome/models/research/object_detection/entrenos/pipelines/efficient_d2_lr1.config"
MODEL_DIR="/home/plome/models/research/object_detection/entrenos/halimeda/efficient_d2/lr1/model_outputs"

# sudo sync
# sudo sysctl -w vm.drop_caches=3
# sudo sync

cp $PIPELINE_CONFIG_PATH $MODEL_DIR/pipeline.config
cp train.sh $MODEL_DIR/train.sh
cd /home/plome/models/research/object_detection/

python3 model_main_tf2.py --model_dir=$MODEL_DIR --num_train_steps=$NUM_TRAIN_STEPS \
  --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
  --pipeline_config_path=$PIPELINE_CONFIG_PATH \
  --checkpoint_every_n=$CHECKPOINT_EVERY_N \
  --num_steps_per_iteration=$NUM_STEPS_PER_ITERATION \
  --checkpoint_max_to_keep=$CHECKPOINT_MAX_TO_KEEP \
  --alsologtostderr

python -c "import time;time.sleep(60);quit()"


PIPELINE_CONFIG_PATH="/home/plome/models/research/object_detection/entrenos/pipelines/efficient_d2_lr2.config"
MODEL_DIR="/home/plome/models/research/object_detection/entrenos/halimeda/efficient_d2/lr2/model_outputs"

cp $PIPELINE_CONFIG_PATH $MODEL_DIR/pipeline.config
cp train.sh $MODEL_DIR/train.sh
cd /home/plome/models/research/object_detection/

python3 model_main_tf2.py --model_dir=$MODEL_DIR --num_train_steps=$NUM_TRAIN_STEPS \
  --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
  --pipeline_config_path=$PIPELINE_CONFIG_PATH \
  --checkpoint_every_n=$CHECKPOINT_EVERY_N \
  --num_steps_per_iteration=$NUM_STEPS_PER_ITERATION \
  --checkpoint_max_to_keep=$CHECKPOINT_MAX_TO_KEEP \
  --alsologtostderr

python -c "import time;time.sleep(60);quit()"

PIPELINE_CONFIG_PATH="/home/plome/models/research/object_detection/entrenos/pipelines/efficient_d2_lr3.config"
MODEL_DIR="/home/plome/models/research/object_detection/entrenos/halimeda/efficient_d2/lr3/model_outputs"

cp $PIPELINE_CONFIG_PATH $MODEL_DIR/pipeline.config
cp train.sh $MODEL_DIR/train.sh
cd /home/plome/models/research/object_detection/

python3 model_main_tf2.py --model_dir=$MODEL_DIR --num_train_steps=$NUM_TRAIN_STEPS \
  --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
  --pipeline_config_path=$PIPELINE_CONFIG_PATH \
  --checkpoint_every_n=$CHECKPOINT_EVERY_N \
  --num_steps_per_iteration=$NUM_STEPS_PER_ITERATION \
  --checkpoint_max_to_keep=$CHECKPOINT_MAX_TO_KEEP \
  --alsologtostderr

