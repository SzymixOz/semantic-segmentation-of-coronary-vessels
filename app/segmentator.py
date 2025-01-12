import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk
import torch
import torchvision
from torchvision import transforms
import numpy as np
import torch.nn as nn
from threading import Thread
from binary_segmentation.skrypt import process_dicom_image
from keypoints.keypoints import get_keypoints
from semantic_segmentation.predict import predict
from semantic_segmentation.converter_RGB import get_colored_mask_with_legend
from save_result import save_result_as_csv


SIZE = 256

class SegmentationApp:
    def __init__(self, root):
        self.root = root
        self.loaded_image = None
        self.photo1 = None
        self.photo2 = None
        self.binary_mask = None
        self.recognized_side = None
        self.detected_side_str = "Detected side:"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.init_gui()

    def init_gui(self):
        self.root.title("Module for automatic segmentation of coronary vessels")
        self.root.geometry("800x400")

        tab1 = ttk.Frame(self.root)

        # Upload image button
        self.upload_button = tk.Button(tab1, text="Upload Image", command=self.upload_image)
        self.upload_button.grid(row=0, column=0, padx=10, pady=10)

        # Predict button
        self.predict_button = tk.Button(tab1, text="Predict", command=self.run_pipeline, state=tk.DISABLED)
        self.predict_button.grid(row=0, column=1, padx=10, pady=10)

        # Empty images for placeholders (store as instance variables)
        self.empty_image_left = ImageTk.PhotoImage(Image.new('RGB', (256, 256), (200, 200, 200)))
        self.empty_image_right = ImageTk.PhotoImage(Image.new('RGB', (470, 256), (200, 200, 200)))

        # Image display labels
        self.image_label1 = tk.Label(tab1, image=self.empty_image_left)
        self.image_label1.grid(row=1, column=0, padx=10, pady=10)

        self.image_label2 = tk.Label(tab1, image=self.empty_image_right)
        self.image_label2.grid(row=1, column=1, padx=10, pady=10)

        # Side recognition label
        self.side_label = tk.Label(tab1, text="")
        self.side_label.grid(row=2, column=0, padx=10, pady=10)

        # Toggle side button
        self.toggle_button = tk.Button(tab1, text="Change side", command=self.toggle_side)
        self.toggle_button.grid_forget()

        tab1.pack(expand=1, fill='both')


    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")])
        if not file_path:
            return

        # Load and display the uploaded image
        self.loaded_image = cv2.imread(file_path)
        resized_image = cv2.resize(self.loaded_image, (SIZE, SIZE))
        image_rgb = cv2.cvtColor(self.loaded_image, cv2.COLOR_BGRA2RGB)
        self.photo1 = ImageTk.PhotoImage(Image.fromarray(resized_image))
        self.image_label1.config(image=self.photo1)
        self.image_label1.image = self.photo1

        # Perform side recognition in a separate thread
        Thread(target=self.run_side_recognition, args=(image_rgb,)).start()

    def run_side_recognition(self, image):
        self.show_processing_message("Performing side recognition...")
        
        self.binary_mask = process_dicom_image(image)
        binary_mask = self.binary_mask
        binary_mask = cv2.resize(binary_mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
        side_model = self.load_side_model()

        # Predict side
        img_tensor = transforms.ToTensor()(binary_mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            side_pred = side_model(img_tensor)
            self.recognized_side = torch.argmax(side_pred).item() == 1

        self.side_label.config(text=f"{self.detected_side_str} {'left' if not self.recognized_side else 'right'}")
        self.predict_button.config(state=tk.NORMAL)
        self.toggle_button.grid(row=2, column=1, padx=10, pady=10)

        self.hide_processing_message()

    def load_side_model(self):
        """Load and prepare the side recognition model."""
        pretrained_net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        side_model = nn.Sequential(
            pretrained_net,
            nn.ReLU(),
            nn.Linear(1000, 2)
        ).to(self.device)
        side_model.load_state_dict(torch.load('side_recognition_model.pth', map_location=self.device))
        side_model.eval()
        return side_model

    def run_pipeline(self):
        """Manages the semantics segmentation pipeline."""
        if self.loaded_image is not None:
            Thread(target=self.pipeline_thread).start()

    def pipeline_thread(self):
        self.show_processing_message("Performing semantic segmantation...")

        binary_mask = self.binary_mask
        keypoints = get_keypoints(binary_mask)
        segmentation_result = self.run_prediction(self.loaded_image, binary_mask, keypoints)
        self.display_and_save_results(segmentation_result)

        self.hide_processing_message()

    def run_prediction(self, img, binary_mask, keypoints):
        """Run the semantic segmentation prediction using UNet model."""
        model_name = 'model_right' if self.recognized_side else 'model_left'
        return predict(img, binary_mask, keypoints, model_name)

    def display_and_save_results(self, segmentation_result):
        """Display the results of the segmentation."""
        result_with_legend = get_colored_mask_with_legend(segmentation_result)
        save_result_as_csv(segmentation_result, "results/segmentation_mask.csv")
        cv2.imwrite("results/segmentation_result.png", cv2.cvtColor(result_with_legend, cv2.COLOR_RGB2BGR))
        self.photo2 = ImageTk.PhotoImage(Image.fromarray(result_with_legend))
        self.image_label2.config(image=self.photo2)
        self.image_label2.image = self.photo2

    def toggle_side(self):
        self.recognized_side = not self.recognized_side
        self.side_label.config(text=f"{self.detected_side_str} {'left' if not self.recognized_side else 'right'}")

    def show_processing_message(self, message):
        self.processing_label = tk.Label(self.root, text=message, fg="red")
        self.processing_label.pack()

    def hide_processing_message(self):
        if hasattr(self, 'processing_label'):
            self.processing_label.pack_forget()


if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationApp(root)
    root.mainloop()
