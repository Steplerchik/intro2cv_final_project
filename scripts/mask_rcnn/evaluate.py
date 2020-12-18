from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn.utils import compute_ap
import numpy as np
from mrcnn.visualize import plot_precision_recall
from utils.utils import compute_recall_prediction
import matplotlib.pyplot as plt


def evaluate_model(dataset, model, cfg, iou_threshold=0.5):
	APs = []
	recall_prediction_list = []
	counter = 0
	for image_id in dataset.image_ids:
		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		scaled_image = mold_image(image, cfg)
		sample = np.expand_dims(scaled_image, 0)
		yhat = model.detect(sample, verbose=0)
		r = yhat[0]
		AP, precisions, recalls, overlaps = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'], iou_threshold)
		APs.append(AP)
		recall, prediction = compute_recall_prediction(r["rois"], gt_bbox, iou_threshold)
		recall_prediction_list.append([recall, prediction])
		counter += 1
		print("Counter:", counter)
	mAP = np.mean(APs)
	return mAP, recall_prediction_list


def get_precision_recall_for_mrcnn(dataset, model, cfg, iou_threshold):
	recall_prediction_list = []
	counter = 0
	for image_id in dataset.image_ids:
		counter += 1
		print("Counter:", counter)
		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		scaled_image = mold_image(image, cfg)
		sample = np.expand_dims(scaled_image, 0)
		yhat = model.detect(sample, verbose=0)
		r = yhat[0]
		recall, prediction = compute_recall_prediction(r["rois"], gt_bbox, iou_threshold)
		recall_prediction_list.append([recall, prediction])

	return recall_prediction_list


def evaluate_model_yolo(dataset, cfg, pred_bboxes, iou_threshold=0.5):
	recall_prediction_list = []
	i = -1
	for image_id in dataset.image_ids:
		i += 1
		print("Counter:", i)
		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		recall, prediction = compute_recall_prediction(pred_bboxes[i], gt_bbox, iou_threshold)
		recall_prediction_list.append([recall, prediction])
	return recall_prediction_list


def get_precision_recall_for_yolo(dataset, cfg, pred_bboxes):
	recall_prediction_list = []
	i = -1
	for image_id in dataset.image_ids:
		i += 1
		print("Counter:", i)
		_, _, _, gt_bbox, _ = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		recall, prediction = compute_recall_prediction(pred_bboxes[i], gt_bbox)
		recall_prediction_list.append([recall, prediction])
		print("Recall:", recall)
		print("Prediction:", prediction)

	return recall_prediction_list


def plot_precision_recall_my(data):
	fig, ax = plt.subplots()
	ax.plot(data[:, 0], data[:, 1])
	ax.set_xlabel('Recall')
	ax.set_ylabel('Precision')
	ax.set_title('Precision x Recall curve')
	ax.grid()
	ax.set_xlim(0, 1.1)
	ax.set_ylim(0, 1.1)
