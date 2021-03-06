from os import listdir, path
from xml.etree import ElementTree
import numpy as np
import imageio as io
import matplotlib.pyplot as plt
from mrcnn.utils import Dataset
import cv2


class ISRLHumanDepthDatasetManager(Dataset):
    def load_dataset(self, dataset_dir, dataset_type="train"):
        self.add_class("dataset", 1, "human")

        images_dir = dataset_dir + '/color/'
        depth_dir = dataset_dir + '/depth/'
        annotations_dir = dataset_dir + '/annotations_voc_xml/'
        image_id_list = []
        image_path_list = []
        depth_path_list = []
        annotation_path_list = []
        for filename in listdir(images_dir):
            image_id = filename[:-4]
            image_path = images_dir + filename
            depth_path = depth_dir + filename
            annotation_path = annotations_dir + image_id + '.xml'

            # bad depth images in the beginning
            if int(image_id) < 16:
                continue

            # Boxes are not in all images
            if not path.exists(annotation_path):
                continue

            image_id_list.append(image_id)
            image_path_list.append(image_path)
            depth_path_list.append(depth_path)
            annotation_path_list.append(annotation_path)

        indices = np.arange(len(image_id_list))
        threshold1 = round(len(image_id_list) * 0.7)
        threshold2 = round(len(image_id_list) * 0.9)
        np.random.seed(1)
        np.random.shuffle(indices)
        if dataset_type == "train":
            indices = indices[:threshold1]
        elif dataset_type == "val":
            indices = indices[threshold1:threshold2]
        else:
            indices = indices[threshold2:]
        print("Indices:", indices)
        for i in indices:
            self.add_image('dataset', image_id=image_id_list[i], path=image_path_list[i],
                           annotation=annotation_path_list[i], depth=depth_path_list[i])

    def get_boxes(self, path):
        root = ElementTree.parse(path).getroot()
        image_boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            image_boxes.append([xmin, ymin, xmax, ymax])

        image_width = int(root.find('.//size/width').text)
        image_height = int(root.find('.//size/height').text)
        return image_boxes, image_width, image_height

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        depth_path = info['depth']
        image_boxes, image_width, image_height = self.get_boxes(path)

        deviation = 300
        kernel = np.ones((10, 10), np.uint16)

        depth_image = io.imread(depth_path)
        depth_image = np.asarray(depth_image)
        boxes_number = len(image_boxes)
        masks = np.zeros([image_height, image_width, boxes_number], dtype='uint8')
        class_ids = []
        for i in range(boxes_number):
            box = image_boxes[i]
            start_row, end_row = box[1], box[3]
            start_column, end_column = box[0], box[2]

            human_area = depth_image[start_row:end_row, start_column:end_column]
            median = np.median(human_area[human_area > 0])
            mask = human_area.copy()
            mask[mask > median + deviation] = 0
            mask[mask < median - deviation] = 0
            mask[mask > 0] = 1
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = mask.astype(np.uint8)
            masks[start_row:end_row, start_column:end_column, i] = mask
            class_ids.append(self.class_names.index('human'))
        return masks, np.asarray(class_ids, dtype='int32')

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
