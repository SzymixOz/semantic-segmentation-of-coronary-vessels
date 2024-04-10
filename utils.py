import os
import zlib
import base64
import cv2
import numpy as np
import pandas as pd
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

def pop_row(df):
    '''Pops the first row of the dataframe and returns both the row and the resulting dataframe.
    Args:
        df (pd.dataframe): Pandas dataframe
    Returns:
        popped_row (pd.dataframe): Pandas dataframe containing only the first row of df
        df (pd.dataframe): Pandas dataframe containing all other rows of df
    '''
    index = df.iloc[:1].index.values[0]
    df_t = df.transpose(copy=True)
    popped_row = df_t.pop(index)
    df = df_t.transpose(copy=True)
    popped_row = popped_row.transpose()
    return popped_row, df

def merge_masks(dataframe, weights={'lekandnow@gmail.com': 1, 'sgurba@gmail.com': 1, 'zamiedzaidalej@gmail.com': 0.7, 'pio.smietanski@gmail.com': 0.7}):
    '''Merges all segmentations from a view of a dataframe using majority voting.
    Args:
        dataframe (pd.dataframe): Pandas dataframe with mandatory columns: user, segmentation
        weights (dict): Dictionary in the format {user: voting weight} (users without a key in the dictionary will be skipped when merging masks)
    Returns:
        Mask obtained by merging all of the masks in dataframe which had a user and weight associated to them. 
    '''
    nr_masks = 1
    row, df = pop_row(dataframe)
    while row['user'] not in list(weights) and not df.empty:
        row, df = pop_row(df)
    mask = unpack_mask(row['segmentation'])
    mask = np.where(mask >= 1, 1.0, 0.0)  # convert to binary mask

    clear_output(wait=True)
    while not df.empty:
        row, df = pop_row(df)
        mask_1 = unpack_mask(row['segmentation'])
        mask_1 = np.where(mask_1 >= 1, 1.0, 0.0)  # convert to binary mask
        if row['user'] in list(weights):
            nr_masks += 1
            # mask_1 *= weights[row['user']]
            mask_1 = mask_1 * weights[row['user']] + (1 - mask_1) * (1 - weights[row['user']]) # should be more correct
            np.add(mask, mask_1, out=mask)
            
    nr_masks = float(nr_masks)
    mask /= nr_masks
    # make round for mask values becasue we want 0 or 1
    mask = np.where(mask >= 0.5, 1.0, 0.0)
    return mask

def get_data(data, voting, images_path):
    '''Given dataframe of segmentation data and local folder of images, returns array of images and segmentations each
    Args:
        data (pd.dataframe): Pandas dataframe with mandatory columns: image_id, frame, segmentation
        voting (bool): Whether to perform majority voting merging of segmentations where possible or randomly choose segmentation
        images_path (string): String containing path to folder where the images are stored
    Returns:
        images (list of np.array): list of images in the form of numpy arrays
        segmentations (list of np.array): list of segmentation masks in the form of numpy arrays
    '''
    images = []
    segmentations = []

    for filename in os.listdir(images_path):
        split = filename.index('_')
        image_id = filename[0:split]
        frame = filename[split+1:-4]

        if voting == False:
            segmentation = data.loc[(data['image_id'] == image_id) & (data['frame'] == int(frame))].sample().iloc[0]['segmentation'] 
            segmentation = unpack_mask(segmentation)
            segmentations.append(segmentation)
            images.append(np.array(Image.open(f'{images_path}/{filename}')))

        if voting == True:
            seg_df = data.loc[(data['image_id'] == image_id) & (data['frame'] == int(frame))]
            if not seg_df.empty:
                mask = merge_masks(seg_df)
                segmentations.append(mask)
                images.append(np.array(Image.open(f'{images_path}/{filename}')))
    return (images, segmentations)

def color_segments(mask):
    segment_colors = {0:[0, 0, 0], 1:[102, 0, 0], 2:[0, 255, 0], 3:[0, 204, 204], 4:[204, 0, 102], 5:[204, 204, 0], 6:[76, 153, 0], 7:[204, 0, 0], 8:[0, 128, 255], 9:[0, 102, 51], 10:[178, 255, 102], 11:[0, 102, 102], 12:[255, 102, 102], 13:[0, 51, 102], 14:[51, 255, 153], 15:[153, 51, 255], 99:[255,255,255], 255:[255, 255, 255]}
    if mask.shape != (512, 512):
        raise Exception("Invalid mask size")
    colored_mask = np.empty((512,512,3), dtype=np.float32)

    it = np.nditer(mask, flags=['multi_index'])
    for pixel in it:
        if pixel.item() not in segment_colors.keys():
            color = 99
        else:
            color = pixel.item()
        for channel in range(0,3):
            colored_mask[it.multi_index[0], it.multi_index[1], channel] = float(segment_colors[color][channel])
    
    return colored_mask

def get_mask(img, mask, binary=False, name=None, img_intensity=0.005, mask_intensity=-0.003):
    """Show segmentation mask on top of original image.

    Args:
        img (np.array): Numpy array containing original image.
        mask (np.array): Numpy array containing binary mask.
        mask_intensity (float, optional): Mask opacity parameter. Defaults to -0.003.
        img_intensity (float, optional): Image opacity parameter. Defaults to 0.005.
    """
    segment_names = {1:"RCA proximal", 2:"RCA mid", 3:"RCA distal", 4:"Posterior descending artery", 5:"Left main", 6:"LAD proximal", 7:"LAD mid", 8:"LAD aplical", 9:"First diagonal", 10:"Second diagonal", 11:"Proximal circumflex artery", 12:"Intermediate/anterolateral artery", 13:"Distal circumflex artery", 14:"Left posterolateral", 15:"Posterior descending", 99:"Unknown"}
    segment_colors = {0:[0, 0, 0], 1:[102, 0, 0], 2:[0, 255, 0], 3:[0, 204, 204], 4:[204, 0, 102], 5:[204, 204, 0], 6:[76, 153, 0], 7:[204, 0, 0], 8:[0, 128, 255], 9:[0, 102, 51], 10:[178, 255, 102], 11:[0, 102, 102], 12:[255, 102, 102], 13:[0, 51, 102], 14:[51, 255, 153], 15:[153, 51, 255], 99:[255,255,255], 255:[255, 255, 255]}
    
    img_color = np.copy(img)
    img_color = cv2.cvtColor(img_color, cv2.COLOR_GRAY2RGB)
    img_color = img_color.astype(np.float32)

    if binary==True:
        # mask = mask > 0.5
        mask_color = cv2.cvtColor(mask.astype(np.float32), cv2.COLOR_GRAY2RGB)
        mask_color[:, :, 1] = 0
        mask_color[:, :, 2] = 0
        mask_color *= 255
    else:
        mask_color = color_segments(mask)

    result = cv2.addWeighted(mask_color, mask_intensity, img_color, img_intensity, 0, img_color)
    if not np.unique(mask).shape[0] < 3:
        segments = np.unique(mask).tolist()[1:]
        handles = []
        for segment in segments:
            if segment not in segment_colors.keys():
                continue
            c = segment_colors[segment]
            color = [x/255 for x in c]
            color = tuple(color)
            handles.append(mpatches.Patch(color=color, label=segment_names[segment]))
            # print(f'segment:{segment}\ncolor:{color}\nname:{segment_names[segment]}')
    # show it with 512x512 resolution
    # Desired output image size
    width, height = 512, 512

    # Create a new figure with the desired size
    plt.figure(figsize=(width / 80, height / 80), dpi=80)
    plt.imshow(result)
    # plt.legend(handles=handles, loc='upper right')
    plt.axis('off')
    if name is not None:
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.savefig(f'./images/{name}.png', bbox_inches='tight', pad_inches=0, transparent=False,dpi=80)

        # Other solution
        # if np.isnan(result).any():
        #     print("Warning: NaN values found in 'result'. Replacing with 0.")
        #     result = np.nan_to_num(result)

        # Check for infinities
        # if np.isinf(result).any():
        #     print("Warning: Infinity values found in 'result'. Replacing with large finite number.")
        #     result = np.nan_to_num(result, posinf=1.0, neginf=0.0)

        # # Check for values outside the range [0, 1]
        # if result.min() < 0 or result.max() > 1:
        #     print("Warning: Values outside the range [0, 1] found in 'result'. Clipping to [0, 1].")
        #     result = np.clip(result, 0, 1)

        # # Now it should be safe to cast to np.uint8
        # result = (result * 255).astype(np.uint8)
        # pil_img = Image.fromarray(result)

        # # Save the image
        # pil_img.save(f'./images/{name}.png')
    return result