import tensorflow as tf
import utilities as uti


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def compute_loss(model, dis_model, x, pre_training):
    mean, log_var = model.encoder(x)
    z = uti.sampling_latent_variable(mean, log_var)

    reconstructed = model.decode(z, True)
    rec_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, reconstructed), axis=(1, 2)))

    KL_divergence = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
    KL_divergence = tf.reduce_mean(tf.reduce_sum(KL_divergence, axis=1))

    batch_size = x.shape[0]
    combined_images = tf.concat([reconstructed, x], axis=0)
    combined_labels = tf.concat([tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0)
    combined_labels += 0.05 * tf.random.uniform(tf.shape(combined_labels))
    discriminator_predicted = dis_model.discriminate(combined_images)
    dis_loss = cross_entropy(combined_labels, discriminator_predicted)
    gen_loss = cross_entropy(tf.ones((batch_size, 1)), dis_model.discriminate(reconstructed))

    # gen_score = tf.sigmoid(dis_model.discriminate(reconstructed))
    # second_term = 0.5 * -(tf.reduce_mean(gen_score - 1.0))
    # first_term = 0.5 * tf.reduce_mean(tf.sigmoid(dis_model.discriminate(x)))

    if pre_training:
        return rec_loss + KL_divergence + 100 * gen_loss, dis_loss, gen_loss
    else:
        clusters = model.group(mean)
        p = uti.compute_t_distribution(mean, alpha=100)
        q = uti.compute_t_distribution(clusters, alpha=1)
        p_q_loss = tf.reduce_sum(-(tf.math.multiply(p, tf.math.log(q))))

        return rec_loss + KL_divergence + 100 * gen_loss + 10 * p_q_loss, dis_loss, gen_loss


@tf.function
def train_step(model, dis_model, x, optimizer, optimizer_dis, pre_training):
    with tf.GradientTape() as tape, tf.GradientTape() as dis_tape:
        loss, dis_loss, _ = compute_loss(model, dis_model, x, pre_training)

    gradients = tape.gradient(loss, model.trainable_variables)
    dis_gradients = dis_tape.gradient(dis_loss, dis_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    optimizer_dis.apply_gradients(zip(dis_gradients, dis_model.trainable_variables))
