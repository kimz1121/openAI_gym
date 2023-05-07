import tensorflow as tf
# import cProfile
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

print(tf.executing_eagerly())
