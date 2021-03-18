import numpy as np

def _get_iou_vector(target, prediction):
    avg_iou = 0.0
    # print("traget shape:", target.shape)
    # print("prediction shape:", prediction.shape)
    for i in range(target.shape[0]):
        target_i = target[i]
        prediction_i = prediction[i]
        intersection = np.logical_and(target_i, prediction_i)
        union = np.logical_or(target_i, prediction_i)
        avg_iou += np.sum(intersection) / np.sum(union)
    #delete variable to save memory
    del intersection, union
    return avg_iou / target.shape[0]

def IoU(y_true, y_pred):
    """Returns Intersection over Union score for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)

