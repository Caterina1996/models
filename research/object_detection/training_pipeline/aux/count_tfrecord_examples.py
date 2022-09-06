
import tensorflow.compat.v1 as tf

filename="test.record"
print(sum(1 for _ in tf.python_io.tf_record_iterator(filename)))