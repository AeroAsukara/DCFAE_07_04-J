
from tensorflow.keras import layers

bn_axis = 3
bn_epsilon = 1.001e-5


def conv_block(x, filters, kernel_size=3, stride=2):
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    x = layers.Activation('relu')(x)

    return x


def de_conv_block(x, filters, kernel_size=3, stride=2):
    x = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, padding='same', strides=stride)(x)
    x = layers.Activation('relu')(x)

    return x


def conv_block_l(x, filters, kernel_size=3, stride=2):
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)

    return x
