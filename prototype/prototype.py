import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import numpy as np
from large_RGB_model import UNet

loaded_image_tab1 = None
photo1_tab1 = None

loaded_image_tab2 = None
photo1_tab2 = None

def upload_image_tab1():
    global loaded_image_tab1, photo1_tab1
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")])
    if file_path:
        loaded_image_tab1 = cv2.imread(file_path)
        resized_image = cv2.resize(loaded_image_tab1, (256, 256))
        resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGRA2RGB)
        photo1_tab1 = ImageTk.PhotoImage(Image.fromarray(resized_image_rgb))
        image_label1_tab1.config(image=photo1_tab1)
        image_label1_tab1.image = photo1_tab1

def predict_image_tab1():
    global loaded_image_tab1
    if loaded_image_tab1 is not None:
        resized_image = cv2.resize(loaded_image_tab1, (256, 256))
        resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_RGBA2RGB)
        resized_image_rgb = predict(resized_image_rgb, 'binary')
        photo2_tab1 = ImageTk.PhotoImage(Image.fromarray(resized_image_rgb))
        image_label2_tab1.config(image=photo2_tab1)
        image_label2_tab1.image = photo2_tab1

def upload_image_tab2():
    global loaded_image_tab2, photo1_tab2
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")])
    if file_path:
        loaded_image_tab2 = cv2.imread(file_path)
        resized_image = cv2.resize(loaded_image_tab2, (256, 256))
        resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGRA2RGB)
        photo1_tab2 = ImageTk.PhotoImage(Image.fromarray(resized_image_rgb))
        image_label1_tab2.config(image=photo1_tab2)
        image_label1_tab2.image = photo1_tab2

def predict_image_tab2():
    global loaded_image_tab2
    if loaded_image_tab2 is not None:
        resized_image = cv2.resize(loaded_image_tab2, (256, 256))
        resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_RGBA2RGB)
        resized_image_rgb = predict(resized_image_rgb, 'dicom')
        photo2_tab2 = ImageTk.PhotoImage(Image.fromarray(resized_image_rgb))
        image_label2_tab2.config(image=photo2_tab2)
        image_label2_tab2.image = photo2_tab2

def predict(img, type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet()
    model = model.to(device)
    if type == 'binary':
        model.load_state_dict(torch.load('large_RGB_model_binary.pth'))
    else:
        model.load_state_dict(torch.load('large_RGB_model_dicom.pth'))
    model.eval()

    print(img.shape)
    print(img)

    img = transforms.ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(img)
        pred = pred.squeeze().cpu().numpy()

    R, G, B = pred[0], pred[1], pred[2]
    pred_image = np.zeros((256, 256, 3), dtype=np.uint8)
    pred_image[..., 0] = (R * 255).astype(np.uint8)
    pred_image[..., 1] = (G * 255).astype(np.uint8)
    pred_image[..., 2] = (B * 255).astype(np.uint8)
    
    return pred_image



root = tk.Tk()
root.title("Semantic Segmentation Prediction Model (prototype)")
root.geometry("600x350")

notebook = ttk.Notebook(root)
notebook.pack(expand=1, fill='both')

tab1 = ttk.Frame(notebook)
notebook.add(tab1, text='binary image prediction')

upload_button_tab1 = tk.Button(tab1, text="Upload Image", command=upload_image_tab1)
upload_button_tab1.grid(row=0, column=0, padx=10, pady=10)

predict_button_tab1 = tk.Button(tab1, text="Predict", command=predict_image_tab1)
predict_button_tab1.grid(row=0, column=1, padx=10, pady=10)

empty_image = ImageTk.PhotoImage(Image.new('RGB', (256, 256), (200, 200, 200)))

image_label1_tab1 = tk.Label(tab1, image=empty_image)
image_label1_tab1.grid(row=1, column=0, padx=10, pady=10)

image_label2_tab1 = tk.Label(tab1, image=empty_image)
image_label2_tab1.grid(row=1, column=1, padx=10, pady=10)

tab2 = ttk.Frame(notebook)
notebook.add(tab2, text='DICOM image prediction')

upload_button_tab2 = tk.Button(tab2, text="Upload Image", command=upload_image_tab2)
upload_button_tab2.grid(row=0, column=0, padx=10, pady=10)

proceed_button_tab2 = tk.Button(tab2, text="Predict", command=predict_image_tab2)
proceed_button_tab2.grid(row=0, column=1, padx=10, pady=10)

image_label1_tab2 = tk.Label(tab2, image=empty_image)
image_label1_tab2.grid(row=1, column=0, padx=10, pady=10)

image_label2_tab2 = tk.Label(tab2, image=empty_image)
image_label2_tab2.grid(row=1, column=1, padx=10, pady=10)

root.mainloop()
