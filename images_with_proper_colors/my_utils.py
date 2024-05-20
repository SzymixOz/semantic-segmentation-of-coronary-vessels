import os
import zlib
import base64
import cv2
import numpy as np
import pandas as pd
import json
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from IPython.display import clear_output


def unpack_mask(mask, shape=(512, 512)):
    """Unpack segmentation mask sent in HTTP request.
    Args:
        mask (bytes): Packed segmentation mask.
    Returns:
        np.array: Numpy array containing segmentation mask.
    """
    mask = base64.b64decode(mask)
    mask = zlib.decompress(mask)
    mask = list(mask)
    mask = np.array(mask, dtype=np.uint8)
    # pylint:disable=too-many-function-args
    mask = mask.reshape(-1, *shape)
    mask = mask.squeeze()
    return mask

def get_data(data, images_path):
    '''Given dataframe of segmentation data and local folder of images, returns array of images and segmentations each
    Args:
        data (pd.dataframe): Pandas dataframe with mandatory columns: image_id, frame, segmentation
        images_path (string): String containing path to folder where the images are stored
    Returns:
        images (list of np.array): list of images in the form of numpy arrays
        segmentations (list of np.array): list of segmentation masks in the form of numpy arrays
    '''
    images = []
    segmentations = []
    filenames = []

    for filename in os.listdir(images_path):

        if data.loc[data['image'] == filename].empty:
            # print(f'{filename} not found in data')
            continue
        
        filenames.append(filename.split('.')[0])

        segmentation = data.loc[data['image'] == filename].sample().iloc[0]['segmentation']
        segmentation = unpack_mask(segmentation)
        segmentations.append(segmentation)
        images.append(np.array(Image.open(f'{images_path}/{filename}')))
    
    return (images, segmentations, filenames)

if __name__ == "__main__":
    data = pd.read_csv('segmentations.csv')
    images, segmentations, filenames = get_data(data, 'images')
