from os import listdir, path
from xml.etree import ElementTree
import numpy as np
from mrcnn.utils import Dataset


class ISRLHumanTestDatasetManager(Dataset):
    def load_dataset(self, dataset_dir, start, size):
        self.add_class("dataset", 1, "human")

        images_dir = dataset_dir + '/color/'
        annotations_dir = dataset_dir + '/annotations_voc_xml/'
        filename_list = []
        for filename in listdir(images_dir):
            filename_list.append(filename)
        filename_list = sorted(filename_list, key=lambda x: int(x[:-4]))
        for filename in filename_list:
            image_id = filename[:-4]
            print(image_id)
            image_path = images_dir + filename
            annotation_path = annotations_dir + image_id + '.xml'

            # bad depth images in the beginning
            if int(image_id) < start or int(image_id) >= start + size:
                continue

            # Boxes are not in all images
            if not path.exists(annotation_path):
                continue

            self.add_image('dataset', image_id=image_id, path=image_path,
                           annotation=annotation_path)

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
        image_boxes, image_width, image_height = self.get_boxes(path)
        boxes_number = len(image_boxes)
        masks = np.zeros([image_height, image_width, boxes_number], dtype='uint8')
        class_ids = []
        for i in range(boxes_number):
            box = image_boxes[i]
            start_row, end_row = box[1], box[3]
            start_column, end_column = box[0], box[2]
            masks[start_row:end_row, start_column:end_column, i] = 1
            class_ids.append(self.class_names.index('human'))
        return masks, np.asarray(class_ids, dtype='int32')

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
