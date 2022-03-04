"""
'model_checkpoint_dir'->'Directory where the saved trained model is stored.')

flags.DEFINE_string('test_dir', '', 'test images folder path.')
flags.DEFINE_string('out_dir', 'results', 'output folder path.')
flags.DEFINE_string('id', '', 'identifier.')
flags.DEFINE_string('n_classes', '', 'number of classes.')
flags.DEFINE_string('labels_file', '', 'path to the class label file')

"""

MODEL_CHECKPOINT_DIR="/home/object/models/research/object_detection/exported_models/v1/saved_model/"

TEST_DIR="/home/object/models/research/object_detection/test_images/halimeda_test"

OUTPUT_DIRECTORY_PATH="/home/object/models/research/object_detection/training/faster_rcnn_inception_resnet_v2/v1/results"

LABELS_FILE="/home/object/models/research/object_detection/data/halimeda_test/label_map.pbtxt"

python3 new_inference.py \
  --model_checkpoint_dir=$MODEL_CHECKPOINT_DIR \
  --test_dir=$TEST_DIR \
  --output_directory=$OUTPUT_DIRECTORY_PATH \
  --labels_file=$LABELS_FILE

#Console output: 

