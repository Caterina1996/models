import os

import tensorflow as tf
from tensorflow import keras
import tensorflow.compat.v1 as tf1
print(tf.version.VERSION)


def print_checkpoint(save_path):
    reader = tf.train.load_checkpoint(save_path)
    print("succes!")
    shapes = reader.get_variable_to_shape_map()
    dtypes = reader.get_variable_to_dtype_map()
    print(f"Checkpoint at '{save_path}':")
    for key in shapes:
        print(f"  (key='{key}', shape={shapes[key]}, dtype={dtypes[key].name}, "f"value={reader.get_tensor(key)})")


def convert_tf2_to_tf1(checkpoint_path, output_prefix):
    """Converts a TF2 checkpoint to TF1.

    The checkpoint must be saved using a 
    `tf.train.Checkpoint(var_list={name: variable})`

    To load the converted checkpoint with `tf.compat.v1.Saver`:
    ```
    saver = tf.compat.v1.train.Saver(var_list={name: variable}) 

    # An alternative, if the variable names match the keys:
    saver = tf.compat.v1.train.Saver(var_list=[variables]) 
    saver.restore(sess, output_path)
    """
    vars = {}
    reader = tf.train.load_checkpoint(checkpoint_path)
    print("correctly loaded!!!")
    dtypes = reader.get_variable_to_dtype_map()
    for key in dtypes.keys():
        # Get the "name" from the 
        if key.startswith('var_list/'):
            var_name = key.split('/')[1]
            # TF2 checkpoint keys use '/', so if they appear in the user-defined name,
            # they are escaped to '.S'.
            var_name = var_name.replace('.S', '/')
            vars[var_name] = tf.Variable(reader.get_tensor(key))

    return tf1.train.Saver().save(sess=None, save_path=output_prefix)
    # return tf1.train.Saver(var_list=vars).save(sess=None, save_path=output_prefix)
# ```
# Convert the checkpoint saved in the snippet `Save a TF2 checkpoint in TF1`:

# ```python
# Make sure to run the snippet in `Save a TF2 checkpoint in TF1`.

# print_checkpoint('tf2-ckpt-saved-in-session-1')
# converted_path = convert_tf2_to_tf1('tf2-ckpt-saved-in-session-1',
#                                     'converted-tf2-to-tf1')
# print("\n[Converted]")
# print_checkpoint(converted_path)

# # Try loading the converted checkpoint.
# with tf.Graph().as_default() as g:
#   a = tf1.get_variable('a', shape=[], dtype=tf.float32, 
#                        initializer=tf1.constant_initializer(0))
#   b = tf1.get_variable('b', shape=[], dtype=tf.float32, 
#                        initializer=tf1.constant_initializer(0))
#   with tf1.variable_scope('scoped'):
#     c = tf1.get_variable('c', shape=[], dtype=tf.float32, 
#                         initializer=tf1.constant_initializer(0))
#   with tf1.Session() as sess:
#     saver = tf1.train.Saver([a, b, c])
#     saver.restore(sess, converted_path)
#     print("\nRestored [a, b, c]: ", sess.run([a, b, c]))




checkpoint_path="/home/object/Desktop/exported_model/no_mines_2k/not_frozen/"
# checkpoint_path="/home/object/caterina/tf_OD_API/models/research/object_detection/entrenos/new_halimeda_test/model_outputs"
model_path = os.path.join(checkpoint_path, "ckpt-19.data-00000-of-00001")

output_prefix="/home/object/Desktop/exported_model/no_mines_2k/out"

print_checkpoint(checkpoint_path)

convert_tf2_to_tf1(checkpoint_path, output_prefix)