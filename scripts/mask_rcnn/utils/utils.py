from mrcnn.utils import compute_overlaps
import numpy as np


def compute_recall_prediction(pred_boxes, gt_boxes, iou=0.5):
    """Compute the recall at the given IoU threshold. It's an indication
    of how many GT boxes were found by the given prediction boxes.

    pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    """
    # Measure overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    if pred_boxes.shape[0] == 0:
        prediction = 0.0
    else:
        prediction = len(set(matched_gt_boxes)) / pred_boxes.shape[0]
    return recall, prediction