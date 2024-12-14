# main.py
import os
import numpy as np
import cv2
from pathlib import Path
import logging

# Import functions from the provided script
from segmentation import preprocess_data, init_model, segment

# Ścieżki katalogów
INPUT_DIR = '../images/images_test/input_dicom'
OUTPUT_DIR = './images_binary/images_test'

def load_dicom_image(file_path):
    ''' Wczytuje obraz DICOM jako macierz numpy i skaluje go do rozmiaru 512x512 '''
    # Wczytanie obrazu DICOM w trybie binarnym
    with open(file_path, 'rb') as f:
        dicom_bytes = f.read()
    print(f"Dicom bytes: {dicom_bytes}")
    
    # Konwersja binarnej zawartości na macierz numpy
    # Za pomocą OpenCV dekodujemy obraz w formacie DICOM
    image = cv2.imdecode(np.frombuffer(dicom_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    print(f"Image shape: {image.shape}")

    # Skaluje obraz do 512x512
    image = cv2.resize(image, (512, 512))
    return image

def save_binary_mask(mask, output_path):
    ''' Zapisuje binarną maskę jako obraz PNG '''
    cv2.imwrite(output_path, mask)

def main():
    logging.basicConfig(level=logging.INFO)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Inicjalizacja modelu
    logging.info('Initializing model...')
    model = init_model()

    # Przetwarzanie obrazów w katalogu wejściowym
    input_files = list(Path(INPUT_DIR).glob('*.png'))
    logging.info(f'Found {len(input_files)} DICOM files in {INPUT_DIR}.')

    for dicom_file in input_files:
        try:
            logging.info(f'Processing file: {dicom_file}')
            
            # Wczytanie obrazu DICOM
            image = load_dicom_image(dicom_file)

            print(image.shape)
            cv2.imshow("Image", image)
            cv2.waitKey(0)
            return
            
            # Segmentacja obrazu
            mask = segment(image, model)
            
            # Ścieżka zapisu maski binarnej
            output_file = Path(OUTPUT_DIR) / (dicom_file.stem + '.png')
            
            # Zapis maski
            save_binary_mask(mask, str(output_file))
            logging.info(f'Mask saved to: {output_file}')
        except Exception as e:
            logging.error(f'Failed to process file {dicom_file}: {e}')

if __name__ == '__main__':
    main()
