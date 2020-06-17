from tensorflow.keras.layers import (
    Input, Conv1D, Conv2D, Dense, BatchNormalization, Dropout, Lambda, Concatenate
)
import tensorflow as tf
from cunet.train.config import config

def tileTensor(s):
    def func(t):
        return tf.tile(t,s)

    return Lambda(func)

def dense_block(
    x, n_neurons, input_dim, initializer, activation='relu'
):
    for i, (n, d) in enumerate(zip(n_neurons, input_dim)):
        extra = i != 0
        x = Dense(n, input_dim=d, activation=activation,
                  kernel_initializer=initializer)(x)
        if extra:
            x = Dropout(0.5)(x)
            x = BatchNormalization(momentum=0.9, scale=True)(x)
    return x


def dense_control(n_conditions, n_neurons):
    """
    For simple dense control:
        - n_conditions = 6
        - n_neurons = [16, 64, 256]
    For complex dense control:
        - n_conditions = 1008
        - n_neurons = [16, 128, 1024]

    """
    input_conditions = Input(shape=(1, config.Z_DIM))
    input_dim = [config.Z_DIM] + n_neurons[:-1]
    initializer = tf.random_normal_initializer(stddev=0.02)
    dense = dense_block(input_conditions, n_neurons, input_dim, initializer)
    gammas = Dense(
        n_conditions, input_dim=n_neurons[-1], activation=config.ACT_G,
        kernel_initializer=initializer
    )(dense)
    betas = Dense(
        n_conditions, input_dim=n_neurons[-1], activation=config.ACT_B,
        kernel_initializer=initializer
    )(dense)
    # both = Add()([gammas, betas])
    return input_conditions, gammas, betas


# DO 2D Condition Change Here
def cnn_block(
    x, n_filters, padding, initializer, activation='relu'
):
    
    kernel_shape = 10

    for i, (f, p) in enumerate(zip(n_filters, padding)):
        extra = i != 0
        x = Conv1D(f, kernel_shape, padding=p, activation=activation,
                   kernel_initializer=initializer)(x)
        if extra:
            x = Dropout(0.5)(x)
            x = BatchNormalization(momentum=0.9, scale=True)(x)
    return x


def cnn_control(n_conditions, n_filters):
    """
    For simple dense control:
        - n_conditions = 6
        - n_filters = [16, 32, 128]
    For complex dense control:
        - n_conditions = 1008
        - n_filters = [16, 32, 64]

    """

    input_conditions = Input(shape=(config.Z_DIM[0], config.Z_DIM[1]))
    initializer = tf.random_normal_initializer(stddev=0.02)
    
    cnn_enc = cnn_block(
        input_conditions, n_filters, config.PADDING, initializer
    )

    gammas_enc = Dense(
        n_conditions, input_dim=n_filters[-1], activation=config.ACT_G,
        kernel_initializer=initializer
    )(cnn_enc)

    gammas_dec = Dense(
        n_conditions, input_dim=n_filters[-1], activation=config.ACT_G,
        kernel_initializer=initializer
    )(cnn_enc)

    betas_enc = Dense(
        n_conditions, input_dim=n_filters[-1], activation=config.ACT_B,
        kernel_initializer=initializer
    )(cnn_enc)

    betas_dec = Dense(
        n_conditions, input_dim=n_filters[-1], activation=config.ACT_B,
        kernel_initializer=initializer
    )(cnn_enc)

    gammas = Concatenate(axis=1)([gammas_enc, gammas_dec])
    betas  = Concatenate(axis=1)([betas_enc, betas_dec])

    # both = Add()([gammas, betas])
    return input_conditions, gammas, betas