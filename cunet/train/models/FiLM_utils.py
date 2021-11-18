import tensorflow as tf
from tensorflow.keras.layers import Lambda

# def FiLM_simple_layer():
#     """multiply scalar to a tensor"""
#     def func(args):
#         x, gamma, beta = args
#         s = list(x.shape)
#         s[0] = 1
#         # avoid tile with the num of batch -> it is the same for both tensors
#         g = tf.tile(tf.expand_dims(tf.expand_dims(gamma, 2), 3), s)
#         b = tf.tile(tf.expand_dims(tf.expand_dims(beta, 2), 3), s)
#         import pdb;pdb.set_trace()
#         return tf.add(b, tf.multiply(x, g))
#     return Lambda(func)
def FiLM_simple_layer():
    """multiply scalar to a tensor"""
    def func(args):
        x, gamma, beta = args
        s = list(x.shape)
        s[0] = 1
        s[2] = 1
        # avoid tile with the num of batch -> it is the same for both tensors

        # print('Gamma Shape: '+str(gamma))
        # print('Beta Shape: '+str(beta))
        # print('Input Shape: '+str(x))

        g = tf.tile(tf.expand_dims(tf.transpose(tf.expand_dims(gamma,-1), [0, 2, 1]), -1), s)
        b = tf.tile(tf.expand_dims(tf.transpose(tf.expand_dims(beta,-1), [0, 2, 1]), -1), s)

        # print('Gamma Shape: '+str(g))
        # print('Beta Shape: '+str(b))

        return tf.add(b, tf.multiply(x, g))
    return Lambda(func)


def FiLM_complex_layer():
    """multiply scalar to a tensor"""
    def func(args):
        x, gamma, beta = args
        s = list(x.shape)
        s[0] = 1
        # avoid tile with the num of batch -> it is the same for both tensors
        # g = tf.tile(tf.expand_dims(gamma, 2), s)
        # b = tf.tile(tf.expand_dims(beta, 2), s)
        g = tf.expand_dims(tf.transpose(gamma, [0, 2, 1]), -1)
        b = tf.expand_dims(tf.transpose(beta, [0, 2, 1]), -1)

        # print(g)

        return tf.add(b, tf.multiply(x, g))
    return Lambda(func)


def slice_tensor(position):
    # Crops (or slices) a Tensor
    def func(x):
        return x[:, :, position]
    return Lambda(func)


def slice_tensor_range(init, end):
    # Crops (or slices) a Tensor
    def func(x):
        return x[:, :, :, init:end]
    return Lambda(func)