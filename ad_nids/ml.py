
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from alibi_detect.models.autoencoder import AE


def build_ae(hidden_dim, encoding_dim, num_hidden, input_dim):

    encoder_net = tf.keras.Sequential()
    encoder_net.add(InputLayer(input_shape=(input_dim,), dtype='float32'))
    for _ in range(num_hidden):
        encoder_net.add(Dense(hidden_dim, activation=tf.nn.relu))
    encoder_net.add(Dense(encoding_dim))

    decoder_net = tf.keras.Sequential()
    decoder_net.add(InputLayer(input_shape=(encoding_dim,)))
    for _ in range(num_hidden):
        decoder_net.add(Dense(hidden_dim, activation=tf.nn.relu))
    decoder_net.add(Dense(input_dim))

    ae = AE(encoder_net, decoder_net)

    return ae
