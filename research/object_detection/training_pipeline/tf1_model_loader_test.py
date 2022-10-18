
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import shutil


MODEL_DIRECTORY_PATH="/home/object/caterina/tf_OD_API/models/research/object_detection/exported_models/no_mines_2k/saved_model"
checkpoint_path="/home/object/caterina/tf_OD_API/models/research/object_detection/exported_models/no_mines_2k/checkpoint"

def load_tf1(path, input):
  print('Loading from', path)
  with tf.Graph().as_default() as g:
    with tf1.Session() as sess:
      meta_graph = tf1.saved_model.load(sess, ["serve"], path)
      sig_def = meta_graph.signature_def[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
      input_name = sig_def.inputs['input'].name
      output_name = sig_def.outputs['output'].name
      print('  Output with input', input, ': ',sess.run(output_name, feed_dict={input_name: input}))



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
  ```
  """
  vars = {}
  reader = tf.train.load_checkpoint(checkpoint_path)
  # detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
  dtypes = reader.get_variable_to_dtype_map()
  for key in dtypes.keys():
    # Get the "name" from the 
    if key.startswith('var_list/'):
      var_name = key.split('/')[1]
      # TF2 checkpoint keys use '/', so if they appear in the user-defined name,
      # they are escaped to '.S'.
      var_name = var_name.replace('.S', '/')
      vars[var_name] = tf.Variable(reader.get_tensor(key))
  
  return tf1.train.Saver(var_list=vars).save(sess=None, save_path=output_prefix)


def load_tf2(path):
  print('Loading from', path)
  model = tf.saved_model.load(path)
  # out = loaded.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY](tf.constant(input))['output']


load_tf2(MODEL_DIRECTORY_PATH)

# detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)


# load_tf1(MODEL_DIRECTORY_PATH, 5.)

# convert_tf2_to_tf1(checkpoint_path, "home/object/Desktop")