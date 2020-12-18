#!/home/nuka-cola/virtual_environments/venv1/bin/python3.6
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import matplotlib.pyplot as plt
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.visualize import apply_mask, random_colors
import numpy as np
from skimage import io
import timeit
from mrcnn.config import Config
import pyrealsense2 as rs
import cv2


class PredictionConfig(Config):
    NAME = "human_cfg"
    NUM_CLASSES = 1 + 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


model_weights_path = 'mask_rcnn_human_cfg_0005_3_train_test_val_human_mask.h5'

cfg = PredictionConfig()
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
model.load_weights('/home/nuka-cola/ultrabot_ws/src/isrl-ultrabot/ultrabot_camera/scripts/mask_rcnn_human_cfg_0005_3_train_test_val_human_mask.h5', by_name=True)

FPS = 15
width = 1280
height = 720

# right_threshold = (width - 756.5) / 1.677
# left_threshold = 427.5 / 1.464
threshold = 300

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, FPS)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

try:
    counter = 0
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        start_time = timeit.default_timer()
        scaled_image = mold_image(color_image, cfg)
        sample = np.expand_dims(scaled_image, 0)
        yhat = model.detect(sample, verbose=0)[0]

        boxes = yhat['rois']
        masks = yhat['masks']
        N = boxes.shape[0]
        distances = [None] * N
        positions = [None] * N
        masked_image = color_image.copy()

        for i in range(N):
            box = boxes[i]
            mask = masks[:, :, i]

            depth_mask = depth_image.copy()
            depth_mask[mask == 0] = 0
            distances[i] = np.median(depth_mask[depth_mask > 0]) * depth_scale
            positions[i] = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

            if positions[i][1] > width / 2:
                angle_tan = (width - positions[i][1]) / distances[i]
            else:
                angle_tan = positions[i][1] / distances[i]

            if angle_tan > threshold:
                masked_image = apply_mask(masked_image, mask, (0.0, 1.0, 0.0))
            else:
                masked_image = apply_mask(masked_image, mask, (0.0, 0.0, 1.0))

        end_time = timeit.default_timer()

        print("New image info:")
        print("Human distances:", distances)
        print("Human positions:", positions)
        print("Detection took: " + str(end_time - start_time) + " secs")

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', masked_image)
        cv2.waitKey(1)
        counter += 1

finally:

    # Stop streaming
    pipeline.stop()
