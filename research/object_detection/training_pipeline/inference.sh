
#INFERENCE


TEST_IMAGES_DIR="/home/object/caterina/tf_OD_API/models/research/object_detection/test_images/halimeda_new_test"
OUTPUT_DIRECTORY_PATH="/home/object/caterina/tf_OD_API/models/research/object_detection/results/new_halimeda_test"
EXPORTED_MODEL_PATH="exported_models/halimeda/FASTER-RCNN/selec_03_10_22/frozen_8k/"
LABELS_FILE="data/halimeda/halimeda_new_data/label_map.pbtxt"

cd /home/object/caterina/tf_OD_API/models/research/object_detection/

python3 inference.py \
  --path_to_images=$TEST_IMAGES_DIR \
  --path_to_out_dir=$OUTPUT_DIRECTORY_PATH \
  --path_to_exported_model=$EXPORTED_MODEL_PATH \
  --path_to_labels=$LABELS_FILE


