import numpy as np
import matplotlib.pyplot as plt
from mrcnn.model import mold_image
from matplotlib.patches import Rectangle
from mrcnn.visualize import display_instances


def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
    for i in range(n_images):
        image = dataset.load_image(i)
        mask, _ = dataset.load_mask(i)
        scaled_image = mold_image(image, cfg)
        sample = np.expand_dims(scaled_image, 0)
        yhat = model.detect(sample, verbose=0)[0]
        plt.subplot(n_images, 2, i * 2 + 1)
        plt.imshow(image)
        plt.title('Actual')
        for j in range(mask.shape[2]):
            plt.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
        plt.subplot(n_images, 2, i * 2 + 2)
        plt.imshow(image)
        plt.title('Predicted')
        ax = plt.gca()
        for box in yhat['rois']:
            y1, x1, y2, x2 = box
            width, height = x2 - x1, y2 - y1
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            ax.add_patch(rect)
    plt.savefig("result", dpi=1000)
    plt.show()


def plot_maskrcnn(dataset, model, cfg, image_id):
    image = dataset.load_image(image_id)
    scaled_image = mold_image(image, cfg)
    sample = np.expand_dims(scaled_image, 0)
    yhat = model.detect(sample, verbose=1)[0]
    display_instances(image, yhat['rois'], yhat['masks'], yhat['class_ids'], ['background', 'human'],
                      yhat['scores'])


def plot_dataset(dataset, image_id):
    image = dataset.load_image(image_id)
    print(image.shape)
    mask, class_ids = dataset.load_mask(image_id)
    print(mask.shape)
    fig, ax = plt.subplots(figsize=[15, 20])
    ax.imshow(image)
    for i in range(mask.shape[2]):
        ax.imshow(mask[:, :, i], cmap='gray', alpha=0.5)
    plt.show()
