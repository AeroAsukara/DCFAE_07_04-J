"""
Instructions.

@author Aero Asukara
"""

import argparse

from controller import orchestrate

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Tensorflow implementation of \'Deep clustering with fusion '
                                                 'autoencoder\', Aero Asukara')
    parser.add_argument('--pretrain', type=bool, default=True,
                        help='If you want to perform the pretrain process (''default: True)')
    parser.add_argument('--epoch', type=int, default=100, help='Epoch control')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size (default: 256)')
    parser.add_argument('--latent_dim', type=int, default=50, help='Latent variable dimension (default: 50)')
    parser.add_argument('--lr_rate', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--dataset', default="mnist", help='Dataset (default: mnist)')
    parser.add_argument('--result', type=int, default=1)
    parser.add_argument('--latent_picture_control', type=int, default=10000)
    parser.add_argument('--loss_compute_control', type=int, default=10000)
    parser.add_argument('--peek', type=bool, default=False,
                        help='If you want get a quick clustering result (''default: False)')

    args = parser.parse_args()
    orchestrate(args)

