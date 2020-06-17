import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, multiply, BatchNormalization, Lambda
)
from tensorflow.keras.optimizers import Adam
from cunet.train.models.FiLM_utils import (
    FiLM_simple_layer, FiLM_complex_layer, slice_tensor, slice_tensor_range
)
from cunet.train.models.control_models import dense_control, cnn_control
from cunet.train.models.unet_model import get_activation, u_net_deconv_block
from cunet.train.config import config


def u_net_conv_block(
    inputs, n_filters, initializer, gamma, beta, activation, film_type,
    kernel_size=(5, 5), strides=(2, 2), padding='same'
):

    x = Conv2D(n_filters, kernel_size=kernel_size,  padding=padding,
               strides=strides, kernel_initializer=initializer)(inputs)
    x = BatchNormalization(momentum=0.9, scale=True)(x)

    x = get_activation(activation)(x)
    return x

def mult(args):
    t1,t2 = args
    return Multiply()([t1, t2])

def set_shape(t):
    s = list(t.shape)
    s[0] = config.BATCH_SIZE
    return tf.reshape(t, s)

def foo(t):
    return t


def cunet_model():
    # axis should be fr, time -> right not it's time freqs
    inputs = Input(shape=config.INPUT_SHAPE)
    n_layers = config.N_LAYERS

    encoder_layers = []
    initializer = tf.random_normal_initializer(stddev=0.02)

    if config.CONTROL_TYPE == 'dense':
        input_conditions, gammas, betas = dense_control(
            n_conditions=config.N_CONDITIONS, n_neurons=config.N_NEURONS)
    if config.CONTROL_TYPE == 'cnn':
        input_conditions, gammas, betas = cnn_control(
            n_conditions=config.N_CONDITIONS, n_filters=config.N_FILTERS)

    if config.FILM_TYPE=='simple':
        inputs_ = FiLM_simple_layer()([inputs, 
            slice_tensor_range(0,int(config.Z_DIM[0]))(gammas), 
            slice_tensor_range(0,int(config.Z_DIM[0]))(betas)])
    elif config.FILM_TYPE=='complex':
        inputs_ = FiLM_complex_layer()([inputs, gammas, betas])

    x = inputs_

    # Encoder
    complex_index = 0
    for i in range(n_layers):
        n_filters = config.FILTERS_LAYER_1 * (2 ** i)

        x = u_net_conv_block(
            x, n_filters, initializer, gammas, betas,
            activation=config.ACTIVATION_ENCODER, film_type=config.FILM_TYPE
        )
        encoder_layers.append(x)
    # Decoder
    for i in range(n_layers):
        # parameters each decoder layer
        is_final_block = i == n_layers - 1  # the las layer is different
        # not dropout in the first block and the last two encoder blocks
        dropout = not (i == 0 or i == n_layers - 1 or i == n_layers - 2)
        # for getting the number of filters
        encoder_layer = encoder_layers[n_layers - i - 1]
        skip = i > 0    # not skip in the first encoder block
        if is_final_block:
            n_filters = 1
            activation = config.ACT_LAST
        else:
            n_filters = encoder_layer.get_shape().as_list()[-1] // 2
            activation = config.ACTIVATION_DECODER
        x = u_net_deconv_block(
            x, encoder_layer, n_filters, initializer, activation, dropout, skip
        ) 

    if config.FILM_TYPE=='simple':
        x_ = FiLM_simple_layer()([x, 
            slice_tensor_range(int(config.Z_DIM[0]),config.Z_DIM[0]*2)(gammas), 
            slice_tensor_range(int(config.Z_DIM[0]),config.Z_DIM[0]*2)(betas)])
    elif config.FILM_TYPE=='complex':
        inputs_ = FiLM_complex_layer()([inputs, gammas, betas])

    outputs = multiply([inputs, x_])

    model = Model(inputs=[inputs, input_conditions], outputs=outputs)
    model.compile(
        optimizer=Adam(lr=config.LR, beta_1=0.5), loss=config.LOSS)
        # experimental_run_tf_function=False)
    return model
