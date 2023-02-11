import numpy as np


def binary_classification_metrics(y_pred, y_true, pos = 1):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    tp = np.size(y_pred[(y_true == y_pred) & (y_pred == pos)])
    fp = np.size(y_pred[(y_true != y_pred) & (y_pred == pos)])
    tn = np.size(y_pred[(y_true == y_pred) & (y_pred != pos)])
    fn = np.size(y_pred[(y_true != y_pred) & (y_pred != pos)])

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = 0

    accuracy = multiclass_accuracy(y_pred, y_true)
    return precision, recall, f1, accuracy

def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    accuracy = (y_pred == y_true).mean()
    return accuracy


def r_squared(y_pred, y_true, force_finite=True):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    non_finite_scores = {1: float('nan'),
                         0: - float('inf')}

    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_total == 0:
        r2 = int(np.array_equal(y_true, y_pred))
    else:
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / ss_total)

    if not force_finite and r2 in (0, 1):
        r2 = non_finite_scores[r2]
    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    mse = np.sum((y_true - y_pred) ** 2) /  np.size(y_true)
    
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    mae = np.sum(abs(y_true - y_pred)) /  np.size(y_true)
    
    return mae
