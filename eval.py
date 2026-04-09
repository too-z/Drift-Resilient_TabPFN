import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def calculate_accuracy(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    return accuracy_score(y_true, y_pred)


def calculate_roc_auc(y_true, y_prob, multi_class='ovr'):
    unique_classes = np.unique(y_true)

    if len(unique_classes) < 2:
        return float('nan')  # ROC AUC undefined for a single class

    if y_prob.shape[1] > 2:
        present_classes = sorted(np.unique(y_true))
       
        if len(present_classes) != y_prob.shape[1]:
            y_prob_filtered = y_prob[:, present_classes]
           
            row_sums = y_prob_filtered.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1e-15
            y_prob_filtered = y_prob_filtered / row_sums
           
            if len(present_classes) == 1:
                return float('nan')
            elif len(present_classes) == 2:
                return roc_auc_score(y_true, y_prob_filtered[:, 1])
            else:
                return roc_auc_score(y_true, y_prob_filtered, multi_class=multi_class, labels=present_classes)
        else:
            return roc_auc_score(y_true, y_prob, multi_class=multi_class)
    else:
        return roc_auc_score(y_true, y_prob[:, 1])


def calculate_f1(y_true, y_prob, average='weighted'):
    y_pred = np.argmax(y_prob, axis=1)
    if y_prob.shape[1] <= 2:
        return f1_score(y_true, y_pred, zero_division=0)
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def calculate_ece(y_true, y_prob, n_bins=15, norm='l1'):
    """
    Computes Calibration Error (Expected, RMS, or Maximum).
    Takes numpy arrays as input (probabilities, not logits).
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    errors, weights = [], []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
       
        if i == n_bins - 1:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)

        n_in_bin = in_bin.sum()
        if n_in_bin == 0:
            continue

        acc_bin = accuracies[in_bin].mean()
        conf_bin = confidences[in_bin].mean()

        errors.append(abs(acc_bin - conf_bin))
        weights.append(n_in_bin)

    if not errors:
        return 0.0

    errors = np.array(errors)
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    if norm == 'l1':
        return float(np.sum(weights * errors))
    elif norm == 'l2':
        return float(np.sqrt(np.sum(weights * errors ** 2)))
    elif norm == 'max':
        return float(np.max(errors))
    else:
        raise ValueError(f"norm must be 'l1', 'l2', or 'max'")


def compute_all_metrics(y_true, y_prob, f1_average='weighted'):
    """Centralized function to return a dictionary of all metrics."""
    return {
        'acc': calculate_accuracy(y_true, y_prob),
        'roc_auc': calculate_roc_auc(y_true, y_prob),
        'f1': calculate_f1(y_true, y_prob, average=f1_average),
        'ece': calculate_ece(y_true, y_prob)
    }

