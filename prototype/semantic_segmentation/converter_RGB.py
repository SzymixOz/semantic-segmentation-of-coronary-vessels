import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from io import BytesIO


SEGMENT_NAMES={
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

SEGMENT_COLORS = {
    '0':   [0],
    '1':   [1],
    '2':   [2],
    '3':   [3],
    '4':   [4],
    '5':   [5],
    '6':   [6],
    '7':   [7],
    '8':   [8],
    '9':   [9],
    '9a':  [10],
    '10':  [11],
    '10a': [12],
    '11':  [13],
    '12':  [14],
    '12a': [15],
    '12b': [16],
    '13':  [17],
    '14':  [18],
    '14a': [29],
    '14b': [20],
    '15':  [21],
    '16':  [22],
    '16a': [23],
    '16b': [24],
    '16c': [25],
    
    '99':  [26],
    '22':  [27],
    '255': [26],
}

COLORS_DICT = {
    0:  [0, 0, 0],        # black
    1:  [128, 0, 0],      # maroon
    2:  [0, 128, 0],      # dark green
    3:  [0, 128, 255],    # medium blue
    4:  [255, 0, 128],    # bright pink
    5:  [255, 255, 0],    # yellow
    6:  [34, 139, 34],    # forest green
    7:  [255, 69, 0],     # orange red
    8:  [70, 130, 180],   # steel blue
    9:  [85, 107, 47],    # olive green
    10: [0, 206, 209],    # turquoise
    11: [154, 205, 50],   # yellow green
    12: [135, 206, 250],  # sky blue
    13: [255, 20, 147],   # deep pink
    14: [250, 128, 114],  # salmon
    15: [255, 165, 0],    # orange
    16: [148, 0, 211],    # dark violet
    17: [75, 0, 130],     # indigo
    18: [220, 20, 60],    # crimson
    19: [210, 105, 30],   # chocolate
    20: [255, 140, 0],    # dark orange
    21: [255, 215, 0],    # gold
    22: [0, 191, 255],    # deep sky blue
    23: [105, 105, 105],  # dim gray
    24: [211, 211, 211],  # light gray
    25: [255, 255, 255],  # white
    26: [199, 21, 133],   # medium violet red
    27: [46, 139, 87],    # sea green
}


# Funkcja do generowania legendy
def generate_legend(mask, segment_names, segment_colors, colors_dict):
    unique_values = np.unique(mask)  # Znalezienie unikalnych wartości w masce
    legend_items = []

    print("Unique values in mask:", unique_values)
    for value in unique_values:
        # Znajdź segment ID na podstawie `segment_colors`
        segment_id = next((key for key, val in segment_colors.items() if value in val), None)
        if segment_id is None:
            name = "Background"
            color = np.array(colors_dict[0]) / 255.0  # Domyślnie czarny dla "Unknown"
        else:
            # Pobierz nazwę segmentu i kolor
            name = segment_names.get(segment_id, "Background")
            color = np.array(colors_dict[value]) / 255.0  # Normalizacja do [0,1]
        
        # Dodaj element do legendy
        # legend_items.append(Patch(color=color, label=f"{name}   (Seg ID: {segment_id}, Seg number: {value})"))
        legend_items.append(Patch(color=color, label=f"{name}"))

    return legend_items


# Główna funkcja
def display_image_with_legend(image, mask, segment_names, segment_colors, colors_dict):
    # Tworzenie legendy
    legend_items = generate_legend(mask, segment_names, segment_colors, colors_dict)

    # Wyświetlanie obrazu z legendą
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title("Prediction Image with Legend")

    # Dodanie legendy z boku
    plt.legend(
        handles=legend_items,
        loc='center left',
        bbox_to_anchor=(1, 0.5),  # Pozycjonowanie legendy obok obrazu
        frameon=False
    )
    plt.show()


# Tworzenie obrazu z maski
def create_colored_image(mask, segment_colors, colors_dict):
    image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for value in np.unique(mask):
        # Znajdź segment ID na podstawie `segment_colors`
        segment_id = next((key for key, val in segment_colors.items() if value in val), None)
        if segment_id is None:
            color = colors_dict[0]  # Domyślnie czarny
        else:
            color = colors_dict[segment_colors[segment_id][0]]  # Pobierz kolor z `colors_dict`

        image[mask == value] = color

    return image


def get_colored_mask(mask):
    return create_colored_image(mask, SEGMENT_COLORS, COLORS_DICT)


def get_colored_mask_with_legend(mask):
    image = create_colored_image(mask, SEGMENT_COLORS, COLORS_DICT)
    legend_items = generate_legend(mask, SEGMENT_NAMES, SEGMENT_COLORS, COLORS_DICT)

    # zwróć obraz do którego z prawej strony będzie przyklejona legenda
    plt.figure(figsize=(3, 3))
    plt.imshow(image)
    plt.axis('off')

    # Dodanie legendy z boku
    plt.legend(
        handles=legend_items,
        loc='center left',
        bbox_to_anchor=(1, 0.5),  # Pozycjonowanie legendy obok obrazu
        frameon=False
    )

    # Zapisz wykres do obiektu w pamięci jako PNG
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')  # Zapis do strumienia
    buffer.seek(0)  # Przesuń wskaźnik strumienia na początek

    # Konwersja bufora do obrazu w formacie numpy (OpenCV)
    np_image = np.frombuffer(buffer.getvalue(), dtype=np.uint8)  # Odczytaj zawartość jako bajty
    buffer.close()  # Zamknij strumień
    cv_image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)  # Dekoduj obraz jako format kolorowy BGR

    # Zwróć obraz w formacie OpenCV
    return cv_image
