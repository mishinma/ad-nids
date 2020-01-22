
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer


def build_net(dims):

    net = tf.keras.Sequential()
    net.add(InputLayer(input_shape=(dims[0],), dtype='float32'))
    for dim in dims[1:]:
        net.add(Dense(dim, activation=tf.nn.relu))

    return net
