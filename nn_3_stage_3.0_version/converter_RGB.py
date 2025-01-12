import numpy as np

colors_dict = {
    0:  [0, 0, 0], # black
    1:  [102, 0, 0], # dark red
    2:  [0, 255, 0], # green
    3:  [0, 204, 204], # light blue
    4:  [204, 0, 102], # pink
    5:  [204, 204, 0], # yellow
    6:  [76, 153, 0], # dark green
    7:  [204, 0, 0], # red
    8:  [0, 128, 255], # blue
    9: [0, 102, 51], # dark green
    10: [0, 102, 102], # light blue
    11: [178, 255, 102], # light green
    12: [178, 255, 202], # light green
    13: [0, 102, 102], # light blue
    14: [255, 102, 102], # light red
    15: [255, 202, 102], # light red
    16: [255, 102, 202], # light red
    17: [0, 51, 102], # dark blue
    18: [51, 255, 153], # light green
    19: [51, 155, 153], # light green
    20: [51, 255, 53], # light green
    21: [153, 51, 255],  # light purple
    22: [255, 255, 0], # yellow
    23: [153, 251, 255], # light blue
    24: [100, 100, 100], # grey
    25: [200, 200, 200], # grey
    26: [255, 255, 255],  # white 
    27: [255, 255, 0], # yellow
    
    # 29: [255, 255, 155], # 
}

# stwórz słownik odwrotny do colors_dict
colors_dict_inv = {tuple(v): k for k, v in colors_dict.items()}

# zrób konwersję zdjęcia z 1 kanału na 3 kanały według poniższej tabeli:
def convert_int_to_RGB(img):
    img = img.astype(np.uint8)
    # zdjęcie jest tablicą dwuwymiarową z wartościami od 0 do 28
    # zamieńmy to na zdjęcie kolorowe
    img_converted = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            img_converted[i, j] = colors_dict[img[i, j]]

    return img_converted

def convert_RGB_to_int(img):
    img = img.astype(np.uint8)
    img_converted = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            img_converted[i, j] = colors_dict_inv[tuple(img[i, j])]

    return img_converted