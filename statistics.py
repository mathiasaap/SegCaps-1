from sklearn.metrics import confusion_matrix
import numpy as np

def calc_precision(stats):
    res = []
    for tp, fp, fn in stats:
        if fp == 0:
            res.append(1)
        else:
            res.append(float(tp)/(tp + fp))
    return res

def calc_recall(stats):
    res = []
    for tp, fp, fn in stats:
        if fn == 0:
            res.append(1)
        else:
            res.append(float(tp)/(tp + fn))
    return res

def calc_sorensen(stats):
    res = []
    for tp, fp, fn in stats:
        if fn == 0 and fp == 0:
            res.append(1)
        else:
            res.append(float(2.0*tp)/(2.0*tp + fp + fn))
    return res

def calc_confusion_matrix(output_label, gt_label, out_classes, labelAxis=None):
    elements = output_label.shape[0]*output_label.shape[1]
    pred_flat = output_label.reshape((-1))
    label_flat = gt_label.reshape((-1))
    matrix = confusion_matrix(label_flat, pred_flat, labels = labelAxis)
    return matrix

def class_stats(matrix):
    classes = matrix.shape[0]

    stats = []

    for c in range(classes):
        tp = matrix[c, c]
        fp = np.sum(matrix[c, :]) - tp
        fn = np.sum(matrix[:, c]) - tp
        stats.append((tp, fp, fn))
    return stats