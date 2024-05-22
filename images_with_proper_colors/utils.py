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


def merge_masks(dataframe, weights={'lekandnow@gmail.com': 1, 'sgurba@gmail.com': 1, 'zamiedzaidalej@gmail.com': 0.7,
                                    'pio.smietanski@gmail.com': 0.7}):
    '''Merges all segmentations from a view of a dataframe using majority voting.
    Args:
        dataframe (pd.dataframe): Pandas dataframe with mandatory columns: user, segmentation
        weights (dict): Dictionary in the format {user: voting weight} (users without a key in the dictionary will be skipped when merging masks)
    Returns:
        Mask obtained by merging all of the masks in dataframe which had a user and weight associated to them. 
    '''
    nr_masks = 1
    row, df = pop_row(dataframe)
    while row['CREATEDBY'] not in list(weights) and not df.empty:
        row, df = pop_row(df)
    mask = unpack_mask(row['segmentation'])
    mask = np.where(mask >= 1, 1.0, 0.0)  # convert to binary mask

    clear_output(wait=True)
    while not df.empty:
        row, df = pop_row(df)
        mask_1 = unpack_mask(row['segmentation'])
        mask_1 = np.where(mask_1 >= 1, 1.0, 0.0)  # convert to binary mask
        if row['CREATEDBY'] in list(weights):
            nr_masks += 1
            # mask_1 *= weights[row['CREATEDBY']]
            mask_1 = mask_1 * weights[row['CREATEDBY']] + (1 - mask_1) * (1 - weights[row['CREATEDBY']])  # should be more correct
            np.add(mask, mask_1, out=mask)

    nr_masks = float(nr_masks)
    mask /= nr_masks
    # print(np.unique(mask))
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
    filenames = []
    labels = []

    print("GETTING DATA...")
    for filename in os.listdir(images_path):
        

        split = filename.index('_')
        image_id = filename[0:split]
        frame = filename[split + 1:-4]

        if data.loc[(data['image_id'] == image_id) & (data['frame'] == int(frame))].empty:
            continue
        
        filenames.append(filename.split(".")[0])

        if voting == False:
            segmentation = data.loc[(data['image_id'] == image_id) & (data['frame'] == int(frame))].sample().iloc[0][
                'segmentation']
            segmentation = unpack_mask(segmentation)
            segmentations.append(segmentation)
            images.append(np.array(Image.open(f'{images_path}/{filename}')))

            json_string = data.loc[(data['image_id'] == image_id) & (data['frame'] == int(frame))].sample().iloc[0]['LABELS']
           
            object_from_json = json.loads(json_string, object_pairs_hook=lambda pairs: {int(key): value for key, value in pairs})
            labels.append(object_from_json)



        if voting == True:
            seg_df = data.loc[(data['image_id'] == image_id) & (data['frame'] == int(frame))]
            if not seg_df.empty:
                mask = merge_masks(seg_df)
                segmentations.append(mask)
                images.append(np.array(Image.open(f'{images_path}/{filename}')))
                # prawdopodobnie trzeba bedzie dodac mergowanie labels dla voting
                json_string = data.loc[(data['image_id'] == image_id) & (data['frame'] == int(frame))].sample().iloc[0]['LABELS']
                object_from_json = json.loads(json_string)
                labels.append(object_from_json)


    return (images, segmentations, filenames, labels)


def color_segments(mask, label, segment_colors):
    if mask.shape != (512, 512):
        raise Exception("Invalid mask size")
    colored_mask = np.empty((512, 512, 3), dtype=np.float32)

    it = np.nditer(mask, flags=['multi_index'])
    for pixel in it:
        if label.get(pixel.item()) in segment_colors.keys():
            color = label.get(pixel.item())
        elif str(pixel.item()) in segment_colors.keys():
            color = str(pixel.item())
        else:
            # in case we dont have certain color for segment (it should not be 0 because it is black color)
            color = '14'
            
        for channel in range(0, 3):
            colored_mask[it.multi_index[0], it.multi_index[1], channel] = float(segment_colors[color][channel])

    return colored_mask


def get_mask(img, mask, label, binary=False, name=None, folder_name='image', img_intensity=0.005, mask_intensity=-0.003, ground_truth=False):
    """Show segmentation mask on top of original image.

    Args:
        img (np.array): Numpy array containing original image.
        mask (np.array): Numpy array containing binary mask.
        mask_intensity (float, optional): Mask opacity parameter. Defaults to -0.003.
        img_intensity (float, optional): Image opacity parameter. Defaults to 0.005.
    """
    # segment_names = {1: "RCA proximal", 2: "RCA mid", 3: "RCA distal", 4: "Posterior descending artery", 5: "Left main",
    #                  6: "LAD proximal", 7: "LAD mid", 8: "LAD aplical", 9: "First diagonal", 10: "Second diagonal",
    #                  11: "Proximal circumflex artery", 12: "Intermediate/anterolateral artery",
    #                  13: "Distal circumflex artery", 14: "Left posterolateral", 15: "Posterior descending",
    #                  99: "Unknown"}
    # print np unique mask
    print(np.unique(mask))
    segment_namess={
        '1': "RCA proximal",
        '2': "RCA mid",
        '3': "RCA distal",
        '4': "Posterior descending artery",
        '5': "Left main",
        '6': "LAD proximal",
        '7': "LAD mid",
        '8': "LAD aplical",
        '9': "First diagonal",
        '9a': "First diagonal a",
        '10': "Second diagonal",
        '10a': "Second diagonal a",
        '11': "Proximal circumflex artery",
        '12': "Intermediate/anterolateral artery",
        '12a': "Obtuse marginal a",
        '12b': "Obtuse marginal b",
        '13': "Distal circumflex artery",
        '14': "Left posterolateral",
        '14a': "Left posterolateral a",
        '14b': "Left posterolateral b",
        '15': "Posterior descending",
        '16': "Posterolateral branch from RCA",
        '16a': "Posterolateral branch from RCA, first",
        '16b': "Posterolateral branch from RCA, second",
        '16c': "Posterolateral branch from RCA, third",
    }
    segment_colors = {
        '0': [0, 0, 0], # black
        '1': [102, 0, 0], # dark red
        '2': [0, 255, 0], # green
        '3': [0, 204, 204], # light blue
        '4': [204, 0, 102], # pink
        # change color for dark brown
        # '5': [204, 102, 0], # dark brown
        '5': [204, 204, 0], # yellow
        '6': [76, 153, 0], # dark green
        '7': [204, 0, 0], # red
        '8': [0, 128, 255], # blue
        '9': [0, 102, 51], # dark green
        '9a':  [0, 102, 102], # light blue
        '10': [178, 255, 102], # light green
        '10a': [178, 255, 202], # light green
        '11': [0, 102, 102], # light blue
        '12': [255, 102, 102], # light red
        '12a':[255, 202, 102], # light red
        '12b': [255, 102, 202], # light red
        '13': [0, 51, 102], # dark blue
        '14': [51, 255, 153], # light green
        '14a':[51, 155, 153], # light green
        '14b':[51, 255, 53], # light green
        '15': [153, 51, 255],  # light purple
        '16':  [255, 255, 0], # yellow
        '16a': [153, 251, 255], # light blue
        '16b':  [100, 100, 100], # grey
        '16c':  [200, 200, 200], # grey
        '99': [255, 255, 255],  # white 
        '22': [255, 255, 0], # yellow 
        '255': [255, 255, 255] # white TODO check if any of these are in any of segmentations
    }

    img_color = np.copy(img)
    img_color = cv2.cvtColor(img_color, cv2.COLOR_GRAY2RGB)
    img_color = img_color.astype(np.float32)

    # print(label)
    if binary == True:
        # mask = mask > 0.5
        mask_color = cv2.cvtColor(mask.astype(np.float32), cv2.COLOR_GRAY2RGB)
        mask_color[:, :, 1] = 0
        mask_color[:, :, 2] = 0
        mask_color *= 255
    else:
        mask_color = color_segments(mask, label, segment_colors)

    result = mask_color / 255.0 if ground_truth else cv2.addWeighted(mask_color, mask_intensity, img_color, img_intensity, 0, img_color)
    #result = cv2.addWeighted(mask_color, mask_intensity, img_color, img_intensity, 0, img_color)
    print(np.unique(mask_color))
    if not np.unique(mask).shape[0] < 3:
        segments = np.unique(mask).tolist()[1:]
        print(segments)
        handles = []
        for segment in segments:
            # if segment not in segment_names.keys():
                # continue
            print(label.get(segment), segment)
            if label.get(segment)=='null' or label.get(segment) is None:
                continue
            c = segment_colors[label.get(segment)]
            color = [x / 255 for x in c]
            color = tuple(color)
            handles.append(mpatches.Patch(color=color, label=segment_namess[label.get(segment)]))
            # print(f'segment:{segment}\ncolor:{color}\nname:{segment_namess[label.get(segment)]}')
    # show it with 512x512 resolution
    # Desired output image size
    width, height = 512, 512

    # Create a new figure with the desired size
    plt.figure(figsize=(width / 80, height / 80), dpi=80)
    plt.imshow(result)
    # plt.legend(handles=handles, loc='upper right')
    plt.axis('off')
    if name is not None:
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(f'./{folder_name}/{name}.png', bbox_inches='tight', pad_inches=0, transparent=False, dpi=80)
        plt.close()
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


if __name__ == "__main__":

    data = pd.read_csv('segmentation_modified.csv', sep=';')
    images2, segmentations2, filenames2, labels2 = get_data(data, voting=True, images_path='./images')
    # for i, (image, seg, filename, label) in enumerate(zip(images2, segmentations2, filenames2, labels2)):
    #     # get_mask(image, seg, label, name=filename, folder_name='segmentation_mask_2', binary=False)
    #     get_mask(image, seg, label, name=filename, folder_name='../keypoints/ground_truth', binary=False, ground_truth=True)
    #     break
