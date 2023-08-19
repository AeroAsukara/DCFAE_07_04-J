import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import h5py
from sklearn.manifold import TSNE
import tensorflow.python.keras.backend as K


"""
DATA AUGMENTATION
"""
data_aux = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=2,
        height_shift_range=2)

data_aux_in = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    channel_shift_range=0.05,
    horizontal_flip=True,
    rescale=0.975,
    zoom_range=[0.95, 1.05]
)


def preprocess_images_mnist(images):
    images = np.expand_dims(images, axis=-1)
    images = images.astype('float32') / 255.
    images = tf.image.resize(images, [32, 32])

    return images


def preprocess_images_cifar(images):
    images = images.astype('float32') / 255.

    return images


def sampling_latent_variable(mean, log_var):
    eps = tf.random.normal(shape=tf.shape(mean))

    return eps * tf.exp(log_var * 0.5) + mean


def usps_load_data(data_path='./Datasets/USPS/usps.h5', data_key="data", target_key="target"):
    with h5py.File(data_path, 'r') as hf:
        train = hf.get('train')
        X_tr = train.get(data_key)[:]
        Y_tr = train.get(target_key)[:]
        test = hf.get('test')
        X_te = test.get(data_key)[:]
        Y_te = test.get(target_key)[:]
        X_tr = X_tr.reshape([-1, 16, 16])
        X_te = X_te.reshape([-1, 16, 16])
        X_tr = np.expand_dims(X_tr, axis=-1)
        X_te = np.expand_dims(X_te, axis=-1)
        X_tr = tf.image.resize(X_tr, [32, 32])
        X_te = tf.image.resize(X_te, [32, 32])

    return (X_tr, Y_tr), (X_te, Y_te)


def coil_20_load_data(data_path='./Datasets/coil_20/coil_20.h5'):
    with h5py.File(data_path, 'r') as hf:
        x = np.asarray(hf.get('data'), dtype='float32')
        y = np.asarray(hf.get('labels'), dtype='int32')
        x = tf.squeeze(x)
        x = np.expand_dims(x, axis=-1)
        x = x.astype('float32') / 255.0
        x = tf.image.resize(x, [32, 32])

    return x, y


def generate_images(model, latent_dim):
    z = tf.random.normal(shape=(16, latent_dim))
    predictions = model.decode(z, True)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i], cmap=plt.cm.binary)
        plt.axis('off')

    plt.show()


def show_original_and_reconstructed_images(model, test_dataset, num_examples_to_generate):
    for test_batch_q in test_dataset.take(2):
        test_sample_q = test_batch_q[0:num_examples_to_generate, :, :, :]

    mean, log_var = model.encoder(test_sample_q)
    z = sampling_latent_variable(mean, log_var)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))
    fig.suptitle('Reconstructed')
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i], cmap=plt.cm.binary)
        plt.axis('off')
    fig_ = plt.figure(figsize=(4, 4))
    fig_.suptitle('Original')
    for i in range(test_sample_q.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(test_sample_q[i], cmap=plt.cm.binary)
        plt.axis('off')

    plt.show()


def show_reconstructed_images(model, test_dataset):
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:16, :, :, :]

    _, _, z, _ = model.encoder(test_sample)
    reconstructed = model.decoder(z)
    fig = plt.figure(figsize=(4, 4))
    fig.suptitle('Reconstructed')
    for i in range(reconstructed.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(reconstructed[i], cmap=plt.cm.binary)
        plt.axis('off')

    fig_ = plt.figure(figsize=(4, 4))
    fig_.suptitle('Original')
    for i in range(test_sample.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(test_sample[i], cmap=plt.cm.binary)
        plt.axis('off')

    plt.show()


def plot_latent_clusters(model, data, labels):
    mean, _ = model.encoder.predict(data)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(mean)
    plt.figure(figsize=(12, 10))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


def plot_clusters(model, data, labels):
    mean, _ = model.encoder.predict(data)
    clusters = model.grouper.predict(mean)
    t_sne = TSNE(n_components=2, perplexity=40, n_iter=300)
    t_sne_results = t_sne.fit_transform(clusters)
    plt.figure(figsize=(12, 10))
    plt.scatter(t_sne_results[:, 0], t_sne_results[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


def plot_label_clusters(model, data, labels):
    mean, _, _, _ = model.encoder.predict(data)
    t_sne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    t_sne_results = t_sne.fit_transform(mean)
    plt.figure(figsize=(12, 10))
    plt.scatter(t_sne_results[:, 0], t_sne_results[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


def plt_trending_curve(axes, trending, title, y_label):
    plt.plot(axes, trending)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()


def compute_t_distribution(h, alpha):
    d = squared_distance(h)
    d = d / alpha
    d = d + 1.0
    d **= (-(alpha + 1.0) / 2.0)
    denominator = tf.reduce_sum(d)
    distribution = d / denominator
    distribution = tf.clip_by_value(distribution, 1e-10, 1.0)

    return distribution


def squared_distance(X, Y=None, W=None):
    if Y is None:
        Y = X

    sum_dimensions = list(range(2, K.ndim(X) + 1))
    X = K.expand_dims(X, axis=1)
    if W is not None:
        D_diag = K.expand_dims(K.sqrt(K.sum(W, axis=1)), axis=1)
        X /= D_diag
        Y /= D_diag
    squared_difference = K.square(X - Y)
    distance = K.sum(squared_difference, axis=sum_dimensions)

    return distance
