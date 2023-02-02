import os
import cv2
import glob
import argparse
import matplotlib
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

from keras.models import load_model
from DenseDepth.layers import BilinearUpSampling2D
from DenseDepth.utils import predict, load_images, display_images
from tensorflow.keras.layers import Layer, InputSpec
from matplotlib import pyplot as plt
from loguru import logger

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='data/test.mp4', type=str, help='Input video.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)
logger.debug('\nModel loaded ({0}).'.format(args.model))

# Input images
cap = cv2.VideoCapture(args.input)
if not cap.isOpened():
    raise Exception("Error while opening video stream.")

logger.info("Video stream opened.")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        frame = np.clip(frame / 255.0, 0, 1).astype(float)
        output = np.squeeze(predict(model, frame, batch_size=1))
        logger.debug(f"{type(output)} - {output.shape}")
        cv2.imshow("Depth map", output)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

