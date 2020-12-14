import numpy as np
import matplotlib.pyplot as plt
from mrcnn.model import mold_image
from matplotlib.patches import Rectangle


def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
	for i in range(n_images):
		image = dataset.load_image(i)
		mask, _ = dataset.load_mask(i)
		scaled_image = mold_image(image, cfg)
		sample = np.expand_dims(scaled_image, 0)
		yhat = model.detect(sample, verbose=0)[0]
		plt.subplot(n_images, 2, i*2+1)
		plt.imshow(image)
		plt.title('Actual')
		for j in range(mask.shape[2]):
			plt.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
		plt.subplot(n_images, 2, i*2+2)
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
