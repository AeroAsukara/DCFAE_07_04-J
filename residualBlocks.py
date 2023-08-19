from tensorflow.keras import layers

bn_axis = 3
bn_epsilon = 1.001e-5


def residual_cell(x, filters, kernel_size=3, stride=1):
    backup = x
    x = layers.Conv2D(filters, kernel_size, stride, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size, stride, padding='same')(x)
    x = layers.Add()([backup, x])
    x = layers.Activation('relu')(x)

    return x


def residual_cell_l(x, filters, kernel_size=3, stride=1):
    backup = x
    x = layers.Conv2D(filters, kernel_size, stride, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(filters, kernel_size, stride, padding='same')(x)
    x = layers.Add()([backup, x])
    x = layers.LeakyReLU(0.2)(x)

    return x


def block(x, filters, kernel_size=3, stride=1, conv_shortcut=True):
    if conv_shortcut:
        short_hand = layers.Conv2D(2 * filters, 1, strides=stride)(x)
    else:
        short_hand = x

    x = layers.Conv2D(filters, 1, strides=stride)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(2 * filters, 1)(x)

    x = layers.Add()([short_hand, x])
    x = layers.LeakyReLU()(x)

    return x


def reverse_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True):
    if conv_shortcut:
        shortcut = layers.Conv2DTranspose(2 * filters, 1, strides=stride)(x)
    else:
        shortcut = x

    x = layers.Conv2DTranspose(filters, 1, strides=stride)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(filters, kernel_size=kernel_size, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(2 * filters, 1)(x)

    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU()(x)

    return x


def stack(x, filters, kernel_size=3, blocks=1, stride=2):
    x = block(x, filters, kernel_size=kernel_size, stride=stride)

    for i in range(2, blocks + 1):
        x = block(x, filters, kernel_size=kernel_size, conv_shortcut=False)

    return x


def reverse_stack(x, filters, kernel_size=3, blocks=1, stride=2):
    x = reverse_block(x, filters, kernel_size=kernel_size, stride=stride)

    for i in range(2, blocks + 1):
        x = reverse_block(x, filters, kernel_size=kernel_size, conv_shortcut=False)

    return x
