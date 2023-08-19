import tensorflow as tf
from tensorflow.keras import layers, Model, Input, regularizers
from residualBlocks import residual_cell
from convNetArchitecture import conv_block, de_conv_block


class CRVAE(Model):
    """
    The encoder and decoder of the FAE framework
    """

    def __init__(self, latent_dim, channels):
        super(CRVAE, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels

        # Encoder
        encoder_inputs = Input(shape=(32, 32, self.channels))
        en = conv_block(encoder_inputs, filters=64)
        en = residual_cell(en, filters=64)
        en = conv_block(en, filters=128)
        en = residual_cell(en, filters=128)
        en = conv_block(en, filters=256)
        en = residual_cell(en, filters=256)
        en = conv_block(en, filters=512)
        en = residual_cell(en, filters=512)
        en = layers.Flatten()(en)
        mean = layers.Dense(self.latent_dim)(en)
        log_var = layers.Dense(self.latent_dim)(en)
        self.encoder = Model(encoder_inputs, [mean, log_var])

        # Decoder
        decoder_inputs = Input(shape=(self.latent_dim, ))
        de = layers.Dense(2 * 2 * 512)(decoder_inputs)
        de = layers.Reshape(target_shape=(2, 2, 512))(de)
        de = layers.Activation('relu')(de)
        de = de_conv_block(de, filters=512)
        de = residual_cell(de, filters=512)
        de = de_conv_block(de, filters=256)
        de = residual_cell(de, filters=256)
        de = de_conv_block(de, filters=128)
        de = residual_cell(de, filters=128)
        de = de_conv_block(de, filters=64)
        de = residual_cell(de, filters=64)
        decoder_outputs = layers.Conv2DTranspose(filters=self.channels, kernel_size=3, padding='same')(de)
        self.decoder = Model(decoder_inputs, decoder_outputs)

        # Deep dense net
        c_inputs = Input(shape=(latent_dim,))
        c = layers.Dense(2048, kernel_regularizer=regularizers.l2(0.001))(c_inputs)
        c = layers.Activation('relu')(c)
        c = layers.Dropout(0.3)(c)
        c = layers.Dense(512, kernel_regularizer=regularizers.l2(0.001))(c)
        c = layers.Activation('relu')(c)
        c = layers.Dropout(0.3)(c)
        c = layers.Dense(512, kernel_regularizer=regularizers.l2(0.001))(c)
        c = layers.Activation('relu')(c)
        c = layers.Dropout(0.3)(c)
        c = layers.Dense(self.latent_dim, kernel_regularizer=regularizers.l2(0.001))(c)
        self.grouper = Model(c_inputs, c)

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))

        return self.decode(eps, apply_sigmoid=True)

    def decode(self, z, apply_sigmoid=False):
        remake = self.decoder(z)
        if apply_sigmoid:
            washed = tf.sigmoid(remake)
            return washed
        return remake

    def group(self, mean):
        clusters = self.grouper(mean)

        return clusters


class Discriminator(Model):
    """
    The Discriminator of the FAE framework
    """

    def __init__(self, channels):
        super(Discriminator, self).__init__()
        self.channels = channels

        discriminator_inputs = Input(shape=(32, 32, self.channels))
        dis = conv_block(discriminator_inputs, filters=32)
        dis = residual_cell(dis, filters=32)
        dis = conv_block(dis, filters=64)
        dis = residual_cell(dis, filters=64)
        dis = conv_block(dis, filters=512)
        dis = residual_cell(dis, filters=512)
        dis = layers.Flatten()(dis)
        discriminator_outputs = layers.Dense(1)(dis)
        self.discriminator = Model(discriminator_inputs, discriminator_outputs)

    @tf.function
    def discriminate(self, x):
        return self.discriminator(x)
