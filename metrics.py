import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score, confusion_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment


def accuracy(true_row_labels, predicted_row_labels):

    cm = confusion_matrix(true_row_labels, predicted_row_labels)
    row_index, column_index = linear_assignment(_make_cost_m(cm))
    total = 0
    for row, column in zip(row_index, column_index):
        value = cm[row][column]
        total += value

    return total * 1. / np.sum(cm)


def _make_cost_m(cm):
    s = np.max(cm)
    return - cm + s


nmi = normalized_mutual_info_score
ari = adjusted_rand_score
acc = accuracy
f1 = f1_score


