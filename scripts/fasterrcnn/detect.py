import numpy as np
import matplotlib.pyplot as plt
import imutils
from imutils import paths
import cv2
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
import argparse
import torch
import imageio
from matplotlib.patches import Rectangle

model = fasterrcnn_resnet50_fpn(pretrained=True).eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", help="path to images directory",
                        default="../../data/color")
    args = vars(parser.parse_args())

    for imagePath in paths.list_images(args["images"]):

        img = imageio.imread(imagePath)
        img = img / img.max()

        pred = model([torch.from_numpy(np.moveaxis(img.astype('float32'), -1, 0))])[0]

        fig, ax = plt.subplots(1, figsize=(20, 20))
        plt.imshow(img)
        plt.axis('off')

        for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
            if label == 1 and score >= 0.5:
                start = box[:2]
                stop = box[2:]
                ax.add_patch(Rectangle(start, *stop - start, facecolor='none', edgecolor='r', linewidth=5))

        plt.show(block=False)
        cv2.waitKey(0)
