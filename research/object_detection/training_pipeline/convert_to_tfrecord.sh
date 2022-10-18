#convert annotated imgs to tfrecord (xml to tfrecord)


# OLD USING MIKI SCRIPTS (IT ALSO WORKS:)

# cd /home/object/caterina/tf_OD_API/models/research/object_detection/scripts
# python xml_to_csv.py --folder /home/object/caterina/tf_OD_API/models/research/object_detection/imgs/halimeda_new_test
# python3 generate_tfrecord.py --folder /home/object/caterina/tf_OD_API/models/research/object_detection/imgs/halimeda_new_test/images
# cd /home/object/caterina/tf_OD_API/models/research/object_detection/imgs/halimeda_new_test/images/
# cp -v *.record /home/object/caterina/tf_OD_API/models/research/object_detection/data/halimeda_new_test

