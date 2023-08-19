import numpy as np
from tensorflow.keras import datasets
import tensorflow as tf
import tensorflow_datasets as tfd
import utilities as uti


def run_coil_100():
    ds = tfd.load('coil100', split='train', batch_size=-1)
    ds = tfd.as_numpy(ds)
    image = ds['image']
    label = ds['object_id']
    image = image.astype('float32') / 255.
    image = tf.image.resize(image, [32, 32])

    cluster_number = 100
    train_size = 7200
    test_size = 7200
    channels = 3
    checkpoint_dir = "D:/Check_Points/coil_100/"

    return image, image, image, label, cluster_number, \
        train_size, test_size, channels, checkpoint_dir


def run_coil_100_():
    ds = tfd.load('coil100', split='train')
    ds = ds.cache()
    ds = ds.batch(256)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    image = list(map(lambda x: x['image'], ds))
    image = map(normalize_img_, image)
    label = list(map(lambda x: x['object_id'], ds))

    cluster_number = 100
    train_size = 7200
    test_size = 7200
    channels = 3
    checkpoint_dir = "D:/Check_Points/coil_100/"

    return image, image, image, label, cluster_number, \
        train_size, test_size, channels, checkpoint_dir


def run_stl_10():
    ds = tfd.load('stl10', split='train', as_supervised=True)
    ds = ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(256)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    ds_test = tfd.load('stl10', split='test', batch_size=-1)
    ds_test = tfd.as_numpy(ds_test)
    test_image = ds_test['image']
    label = ds_test['label']
    test_image = test_image.astype('float32') / 255.

    random_index_label = np.random.randint(0, 8000, size=100)
    test_image = tf.gather(test_image, random_index_label)
    label = label[random_index_label]

    cluster_number = 10
    train_size = 5000
    test_size = 100
    channels = 3
    checkpoint_dir = "D:/Check_Points/slt10/"

    return ds, test_image, test_image, label, cluster_number, \
        train_size, test_size, channels, checkpoint_dir


# def normalize_img(image, label):
#     return tf.image.resize(tf.cast(image, tf.float32) / 255., [32, 32]), label

def normalize_img(image, label):

    return tf.cast(image, tf.float32) / 255., label


def normalize_img_(image):
    return tf.cast(image, tf.float32) / 255.


def run_stl_10_train():
    ds = tfd.load('stl10', split='unlabelled', as_supervised=True)
    ds = ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(256)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    ds_test = tfd.load('stl10', split='test', batch_size=-1)
    ds_test = tfd.as_numpy(ds_test)
    test_image = ds_test['image']
    label = ds_test['label']
    test_image = test_image.astype('float32') / 255.

    random_index_label = np.random.randint(0, 8000, size=100)
    test_image = tf.gather(test_image, random_index_label)
    label = label[random_index_label]

    cluster_number = 10
    train_size = 20000
    test_size = 100
    channels = 3
    checkpoint_dir = "D:/Check_Points/slt10/"

    return ds, test_image, test_image, label, cluster_number, \
        train_size, test_size, channels, checkpoint_dir


def run_mnist():
    (train_images_mnist, train_labels_mnist), (test_images_mnist, test_labels_mnist) = datasets.mnist.load_data()
    train_images = uti.preprocess_images_mnist(train_images_mnist)
    test_images = uti.preprocess_images_mnist(test_images_mnist)
    image_data = np.concatenate((train_images, test_images))
    label_data = np.concatenate((train_labels_mnist, test_labels_mnist))
    cluster_number = 10
    train_size = 70000
    test_size = 10000

    channels = 1
    checkpoint_dir = "D:/Check_Points/mnist_ablation/"

    return image_data, test_images, test_images, label_data, cluster_number, \
        train_size, test_size, channels, checkpoint_dir


def run_mnist_t():
    (_, _), (test_images_mnist, test_labels_mnist) = datasets.mnist.load_data()
    test_images = uti.preprocess_images_mnist(test_images_mnist)
    cluster_number = 10
    train_size = 10000
    test_size = 10000
    channels = 1
    checkpoint_dir = "D:/Check_Points/mnist_ablation_t/"

    return test_images, test_images, test_images, test_labels_mnist, cluster_number, \
        train_size, test_size, channels, checkpoint_dir


def run_fashion_mnist():
    (train_images_fashion_mnist, train_labels_fashion_mnist), (test_images_fashion_mnist, test_labels_fashion_mnist) = \
        datasets.fashion_mnist.load_data()
    train_images = uti.preprocess_images_mnist(train_images_fashion_mnist)
    test_images = uti.preprocess_images_mnist(test_images_fashion_mnist)
    image_data = np.concatenate((train_images, test_images))
    label_data = np.concatenate((train_labels_fashion_mnist, test_labels_fashion_mnist))

    cluster_number = 10
    train_size = 70000
    test_size = 10000
    channels = 1
    checkpoint_dir = "D:/Check_Points/fashion_mnist_ablation/"

    return image_data, test_images, test_images, label_data, cluster_number, \
        train_size, test_size, channels, checkpoint_dir


def run_usps():
    (train_images, train_labels), (test_images, label_for_plot_latent_space) = uti.usps_load_data()
    image_data = np.concatenate((train_images, test_images))
    label_data = np.concatenate((train_labels, label_for_plot_latent_space))
    images_for_plot_latent_space = image_data
    cluster_number = 10
    train_size = 9298
    test_size = 9298
    channels = 1
    checkpoint_dir = "D:/Check_Points/usps_ablation/"

    return image_data, image_data, images_for_plot_latent_space, label_data, cluster_number, \
        train_size, test_size, channels, checkpoint_dir


def run_coil_20():
    train_images, label_for_plot_latent_space = uti.coil_20_load_data()
    images_for_plot_latent_space = train_images
    test_images = train_images
    cluster_number = 20
    train_size = 1440
    test_size = 1440
    channels = 1
    checkpoint_dir = "D:/Check_Points/COIL_20/"

    return train_images, test_images, images_for_plot_latent_space, label_for_plot_latent_space, cluster_number, \
        train_size, test_size, channels, checkpoint_dir


def run_cifar_10():
    (train_images_cifar_10, train_labels_cifar_10), (test_images_cifar_10, test_labels_cifar_10) = datasets. \
        cifar10.load_data()
    train_images = uti.preprocess_images_cifar(train_images_cifar_10)
    test_images = uti.preprocess_images_cifar(test_images_cifar_10)
    image_data = np.concatenate((train_images, test_images))
    random_index = np.random.randint(0, 60000, size=100)
    test_images = image_data[random_index]

    label_data = np.concatenate((train_labels_cifar_10, test_labels_cifar_10))
    label_data = label_data[random_index]
    label_data = np.reshape(label_data, label_data.shape[0])

    # images_for_plot_latent_space = test_images
    # label_for_plot_latent_space = np.reshape(test_labels_cifar_10, test_labels_cifar_10.shape[0])
    cluster_number = 10
    train_size = 60000
    test_size = 60000
    channels = 3
    checkpoint_dir = "D:/Check_Points/cifar_10_ablation/"

    return image_data, test_images, test_images, label_data, cluster_number, \
        train_size, test_size, channels, checkpoint_dir


options = {
    'mnist': run_mnist,
    'mnist_t': run_mnist_t,
    'fashion_mnist': run_fashion_mnist,
    'usps': run_usps,
    'cifar_10': run_cifar_10,
    'coil_20': run_coil_20,
    'coil_100': run_coil_100,
    'stl_10': run_stl_10,
    'stl_train': run_stl_10_train,
}
