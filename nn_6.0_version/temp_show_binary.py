# wczytaj zdjęcie z pliku '/images/images_train'
# wylistuj wszystkie pliki w katalogu '/images/images_train'
import cv2
import os

images_path = './image_binary'
images = os.listdir(images_path)

for image in images:
    image_path = os.path.join(images_path, image)
    img = cv2.imread(image_path)
    # print(img*100)
    # wywietl zdjęcie
    cv2.imshow('image', img*100)
    cv2.waitKey(0)

