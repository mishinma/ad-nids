
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from alibi_detect.models.autoencoder import AE


def build_ae(hidden_dim, encoding_dim, num_hidden, input_dim):

    encoder_net = tf.keras.Sequential(
        [InputLayer(input_shape=(input_dim,))] +
        [Dense(hidden_dim, activation=tf.nn.relu)]*num_hidden,
        [Dense(encoding_dim)]
    )

    decoder_net = tf.keras.Sequential(
        [InputLayer(input_shape=(encoding_dim,))] +
        [Dense(hidden_dim, activation=tf.nn.relu)] * num_hidden,
        [Dense(input_dim)]
    )

    ae = AE(encoder_net, decoder_net)

    return ae
