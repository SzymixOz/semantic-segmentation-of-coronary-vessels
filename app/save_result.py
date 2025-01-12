import cv2
import numpy as np
import csv
from semantic_segmentation.converter_RGB import SEGMENT_COLORS


def save_result_as_csv(segmentation_result, output_csv_path):
    """Save the segmentation result as a CSV file."""
    resized_result = cv2.resize(segmentation_result, (512, 512), interpolation=cv2.INTER_NEAREST)

    # Odwrócenie mapy: wartość -> klucz
    value_to_key = {v[0]: k for k, v in SEGMENT_COLORS.items()}

    # Konwersja wartości na odpowiadające nazwy
    mapped_result = np.empty_like(resized_result, dtype=object)
    for value, key in value_to_key.items():
        mapped_result[resized_result == value] = key

    # Zapis do CSV
    with open(output_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(mapped_result)

    print(f"Mask saved to {output_csv_path}")
