
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer


def build_net(input_shape, hidden_dims, activations):
    if len(hidden_dims) != len(activations):
        raise ValueError('The number of hidden layers must be'
                         ' the same as the number of activations')
    net = tf.keras.Sequential()
    net.add(InputLayer(input_shape=input_shape))
    for dim, activation in zip(hidden_dims, activations):
        net.add(Dense(dim, activation=activation))

    return net
