import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
import keras
import numpy as np
import logging

def preprocess_data(image):
    ''' Turns lists of images and segmentations (represented as numpy arrays) into arrays with the right shape for the U-net model
    Args:
        images (list of np.array): list of images in the form of numpy arrays
    Returns:
        X (np.array): numpy array of inputs in the right shape for U-net model
    '''
    X = np.array(image).astype(float)
    assert X.shape == (512,512), 'image should be 512x512'
    X = np.repeat(X[..., np.newaxis], 3, -1)
    X = np.expand_dims(X, axis=0)
    return X


def init_model(backbone='densenet121', weights_path='segmentation_weights.h5'):
    ''' Initializes U-net model and loads pretrained weights
    Args:
        backbone (string): encoder to be used
        weights_path (string): path to pretrained weights
    Returns:
        model (segmentation_models.model)
    '''
    loss = sm.losses.DiceLoss()
    model_metrics = [sm.metrics.Precision(), sm.metrics.Recall(), sm.metrics.FScore()]
    model = sm.Unet(backbone, encoder_weights='imagenet', input_shape=(None, None, 3), classes=2, activation='sigmoid')
    logging.getLogger().debug('Segmentation model initialized')
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer, loss, metrics=model_metrics)
    logging.getLogger().debug('Segmentation model compiled')
    model.load_weights(weights_path)
    return model


def get_preprocessing(backbone):
    ''' Gets preprocessing function corresponding to encoder
    Args:
        backbone (string): encoder of which to get preprocessing
    Returns:
        preprocess_input (function): function to be applied to inputs before feeding into the model
    '''
    preprocess_input = sm.get_preprocessing(backbone)
    return preprocess_input


def get_segmentation(model, image):
    ''' Predicts binary semantic segmentation of preprocessed image
    Args:
        model (segmentation_models.model): model to be used in predictions
        image (numpy.array): numpy array of image (after applying preprocessing)
    Returns:
        mask (numpy.array): binary semantic segmentation corresponding to image (0-background, 1-vessel)
    '''
    mask = model(image)
    mask = np.array(mask)
    mask = (mask[0, :, :, 1] > 0.5).astype(np.uint8)
    return mask


def segment(image, model, backbone='densenet121'):
    ''' Predicts binary semantic segmentation of raw image
    Args:
        image (numpy.array): numpy array of image
    Returns:
        mask (numpy.array): binary semantic segmentation corresponding to image (0-background, 1-vessel)
    '''
    preprocess_input = sm.get_preprocessing(backbone)
    
    try:
        input = preprocess_data(image)
    except Exception as e:
        logging.getLogger().error('Error while preprocessing image for segmentation:', exc_info=e)

    input = preprocess_input(input)

    try:
        mask = get_segmentation(model, input)
    except Exception as e:
        logging.getLogger().error('Error predicting segmentation mask:', exc_info=e)

    return mask

import cv2
if __name__ == '__main__':
    model = init_model()
    folder_source = "../images_with_proper_colors/images/"
    folder_destination = './images_binary/'
    for image in os.listdir(folder_source):
        image_mask = cv2.imread(folder_source + image, cv2.IMREAD_GRAYSCALE)
        mask = segment(image_mask, model)
        cv2.imwrite(folder_destination + image, mask*255)

    

