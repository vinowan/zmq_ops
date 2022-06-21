import os.path
import tensorflow as tf

so_file = os.path.join(os.path.dirname(__file__), "libzmq_ops.so")
zmq_ops = tf.load_op_library(so_file)