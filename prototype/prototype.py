import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


def upload_image():
    global loaded_image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")])
    if file_path:
        loaded_image = Image.open(file_path)
        resized_image = loaded_image.resize((250, 250))
        photo1 = ImageTk.PhotoImage(resized_image)
        image_label1.config(image=photo1)
        image_label1.image = photo1

def predict_image():
    global loaded_image
    if loaded_image:
        resized_image = loaded_image.resize((250, 250))
        photo2 = ImageTk.PhotoImage(resized_image)
        # call model prediction
        image_label2.config(image=photo2)
        image_label2.image = photo2


root = tk.Tk()
root.title("Image Uploader")
root.geometry("600x350")

loaded_image = None

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.grid(row=0, column=0, padx=10, pady=10)

predict_button = tk.Button(root, text="Predict", command=predict_image)
predict_button.grid(row=0, column=1, padx=10, pady=10)

empty_image = ImageTk.PhotoImage(Image.new('RGB', (250, 250), (200, 200, 200)))

image_label1 = tk.Label(root, image=empty_image)
image_label1.grid(row=1, column=0, padx=10, pady=10)

image_label2 = tk.Label(root, image=empty_image)
image_label2.grid(row=1, column=1, padx=10, pady=10)

root.mainloop()
