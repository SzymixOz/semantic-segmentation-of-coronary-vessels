import numpy as np
import cv2
import logging
from .segmentation import segment, init_model


def process_dicom_image(image):
    try:
        # if image.shape != (512, 512):
        #     raise ValueError(f"Image must have dimensions 512x512, but got {image.shape}")
        logging.getLogger().info(f"Processing image with shape {image.shape}")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (512, 512))
        logging.getLogger().info(f"Processed image shape: {image.shape}")
        model = init_model(backbone='densenet121', weights_path='./binary_segmentation/segmentation_weights.h5')
        mask = segment(image, model, backbone='densenet121')

        return mask

    except Exception as e:
        logging.getLogger().error(f"Error during binary segmentation: {e}")
        return None
