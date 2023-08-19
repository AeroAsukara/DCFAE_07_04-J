import numpy as np
import tensorflow as tf
import time
import metrics
import utilities as uti
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from IPython import display
from core import CRVAE, Discriminator
from train import train_step, compute_loss
from gateway import options


def orchestrate(args):
    """
    PARAMETERS
    """
    pre_train = args.pretrain
    if pre_train:
        cluster_trainable = False
        epoch_control = args.epoch
        result_control = args.result
        latent_picture_control = args.latent_picture_control
        check_point_control = epoch_control
        loss_compute_control = args.loss_compute_control
    else:
        cluster_trainable = True
        epoch_control = args.epoch
        result_control = args.result
        latent_picture_control = args.latent_picture_control
        check_point_control = 500000000
        loss_compute_control = args.loss_compute_control

    batch_size = args.batch_size
    latent_dim = args.latent_dim
    lr_rate = args.lr_rate

    """
    DATASET IDENTIFICATION
    """
    train_images, test_images, images_for_plot_latent_space, label_for_plot_latent_space, cluster_number, \
        train_size, test_size, channels, checkpoint_dir = options[args.dataset]()

    """
    MAIN
    """
    if args.dataset not in {'stl_train', 'stl_10'}:
        train_dataset = (tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_size)
                         .batch(batch_size))
    else:
        train_dataset = list(map(lambda x: x[0], train_images))

    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_size)
                    .batch(batch_size))

    epochs = epoch_control
    num_examples_to_generate = 16

    random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])

    optimizer = tf.keras.optimizers.Adam(lr_rate)
    optimizer_dis = tf.keras.optimizers.Adam(lr_rate)
    model = CRVAE(latent_dim, channels)
    model.grouper.trainable = cluster_trainable
    dis_model = Discriminator(channels)
    checkpoint = tf.train.Checkpoint(model=model,
                                     dis_model=dis_model,
                                     optimizer=optimizer,
                                     optimizer_dis=optimizer_dis)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    x_axis = np.empty(0)
    metrics_x_axis = np.empty(0)
    fusion_loss_history = np.empty(0)
    acc_history = np.empty(0)
    nmi_history = np.empty(0)
    ari_history = np.empty(0)
    dis_loss_history = np.empty(0)
    gen_loss_history = np.empty(0)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:
            if pre_train is not True:
                train_x = uti.data_aux.flow(train_x, batch_size=batch_size).next()

            train_step(model, dis_model, train_x, optimizer, optimizer_dis, pre_train)

        end_time = time.time()

        if (epoch % result_control) == 0:
            metrics_x_axis = np.append(metrics_x_axis, epoch)
            mean, _ = model.encoder.predict(test_images)
            clusters = model.group(mean)
            # predicted_cluster_centers = KMeans(n_clusters=cluster_number,
            #                                    n_init=20,
            #                                    algorithm='lloyd',
            #                                    random_state=10).fit_predict(clusters)

            predicted_cluster_centers = SpectralClustering(n_clusters=cluster_number,
                                                           n_init=20,
                                                           ).fit_predict(clusters)

            nmi = np.round(metrics.nmi(label_for_plot_latent_space, predicted_cluster_centers), 5)
            acc = np.round(metrics.acc(label_for_plot_latent_space, predicted_cluster_centers), 5)
            ari = np.round(metrics.ari(label_for_plot_latent_space, predicted_cluster_centers), 5)
            print('Epoch: {}, ACC: {:.5}, NMI: {:.5}, ARI: {:.5} '.format(epoch, acc, nmi, ari))
            acc_history = np.append(acc_history, acc)
            nmi_history = np.append(nmi_history, nmi)
            ari_history = np.append(ari_history, ari)

        loss = tf.keras.metrics.Mean()
        dis_loss = tf.keras.metrics.Mean()
        gene_loss = tf.keras.metrics.Mean()

        if (epoch % loss_compute_control) == 0:
            x_axis = np.append(x_axis, epoch)
            for test_x in test_dataset:
                fusion_loss, related_loss, gen_loss = compute_loss(model, dis_model, test_x, pre_training=pre_train)
                loss(fusion_loss)
                dis_loss(related_loss)
                gene_loss(gen_loss)

            main_loss = loss.result()
            dis_loss = dis_loss.result()
            gen_loss = gene_loss.result()
            fusion_loss_history = np.append(fusion_loss_history, main_loss)
            dis_loss_history = np.append(dis_loss_history, dis_loss)
            gen_loss_history = np.append(gen_loss_history, gen_loss)
            display.clear_output(wait=False)
            print('Epoch: {}, Test set, Fusion loss: {:.5}, Dis loss: {:.3}, Gen loss: {:.3}, Time used: {:.3} '
                  .format(epoch, main_loss, dis_loss, gen_loss, end_time - start_time))

        if (epoch + 1) % check_point_control == 0:
            checkpoint.save(checkpoint_dir)

        if (epoch + 1) % latent_picture_control == 0:
            uti.plot_latent_clusters(model, images_for_plot_latent_space, label_for_plot_latent_space)
            uti.plot_clusters(model, images_for_plot_latent_space, label_for_plot_latent_space)

    uti.plt_trending_curve(x_axis, fusion_loss_history, 'Fusion Loss Trending', 'Fusion Loss')
    uti.plt_trending_curve(x_axis, dis_loss_history, 'Discriminator Loss Trending', 'Discriminator Loss')
    uti.plt_trending_curve(x_axis, gen_loss_history, 'Generator Loss Trending', 'Generator Loss')
    uti.plt_trending_curve(metrics_x_axis, acc_history, 'ACC Trending', 'ACC')
    uti.plt_trending_curve(metrics_x_axis, nmi_history, 'NMI Trending', 'NMI')
    uti.plt_trending_curve(metrics_x_axis, ari_history, 'ARI Trending', 'ARI')
    uti.generate_images(model, latent_dim)
    uti.show_original_and_reconstructed_images(model, test_dataset, num_examples_to_generate)