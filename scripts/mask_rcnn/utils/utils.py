from mrcnn.model import load_image_gt
from mrcnn.utils import compute_overlaps
import numpy as np
from mrcnn.model import mold_image
import matplotlib.pyplot as plt


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


def find_ground_truth_number(dataset, cfg):
    ground_truth_number = 0
    for image_id in dataset.image_ids:
        _, _, _, gt_bboxes, _ = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        ground_truth_number += len(gt_bboxes)
    return ground_truth_number


def find_TPs_FPs(dataset, model, cfg, iou_threshold=0.5):
    tps_fps = np.empty((0, 3))
    counter = 0
    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        scaled_image = mold_image(image, cfg)
        sample = np.expand_dims(scaled_image, 0)
        yhat = model.detect(sample, verbose=0)
        r = yhat[0]
        overlaps = compute_overlaps(r["rois"], gt_bbox)
        iou_max = np.max(overlaps, axis=1)
        for i in range(len(iou_max)):
            if iou_max[i] > iou_threshold:  # TP
                add_data = np.array([r["scores"][i], 1, 0])
            else:  # FP
                add_data = np.array([r["scores"][i], 0, 1])

            tps_fps = np.vstack((tps_fps, add_data))

        print("Counter:", counter)
        counter += 1
    return tps_fps


def sort_TPs_FPs(tps_fps):
    return np.flip(tps_fps[tps_fps[:, 0].argsort()], axis=0)


def compute_precision_recall_CORRECTLY(tps_fps, ground_truth):
    acc_TP = 0
    acc_FP = 0
    precision_recall = np.empty((0, 2))
    for bbox_info in tps_fps:
        acc_TP += bbox_info[1]
        acc_FP += bbox_info[2]
        precision = acc_TP / (acc_TP + acc_FP)
        recall = acc_TP / ground_truth
        precision_recall = np.vstack((precision_recall, np.array([precision, recall])))
    return precision_recall


def plot_precision_recall_CORRECTLY(data):
    fig, ax = plt.subplots()
    ax.plot(data[:, 1], data[:, 0])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision x Recall curve')
    ax.grid()
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)


def get_precision_recall(dataset, model, cfg, iou_threshold=0.5):
    ground_truth = find_ground_truth_number(dataset, cfg)
    tps_fps = find_TPs_FPs(dataset, model, cfg, iou_threshold)
    tps_fps_sorted = sort_TPs_FPs(tps_fps)
    precision_recall = compute_precision_recall_CORRECTLY(tps_fps_sorted, ground_truth)
    return precision_recall
