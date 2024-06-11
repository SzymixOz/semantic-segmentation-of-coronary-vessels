import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

loaded_image_tab1 = None
photo1_tab1 = None

loaded_image_tab2 = None
photo1_tab2 = None

def upload_image_tab1():
    global loaded_image_tab1, photo1_tab1
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")])
    if file_path:
        loaded_image_tab1 = Image.open(file_path)
        resized_image = loaded_image_tab1.resize((250, 250))
        photo1_tab1 = ImageTk.PhotoImage(resized_image)
        image_label1_tab1.config(image=photo1_tab1)
        image_label1_tab1.image = photo1_tab1

def predict_image_tab1():
    global loaded_image_tab1
    if loaded_image_tab1:
        resized_image = loaded_image_tab1.resize((250, 250))
        photo2_tab1 = ImageTk.PhotoImage(resized_image)
        image_label2_tab1.config(image=photo2_tab1)
        image_label2_tab1.image = photo2_tab1

def upload_image_tab2():
    global loaded_image_tab2, photo1_tab2
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")])
    if file_path:
        loaded_image_tab2 = Image.open(file_path)
        resized_image = loaded_image_tab2.resize((250, 250))
        photo1_tab2 = ImageTk.PhotoImage(resized_image)
        image_label1_tab2.config(image=photo1_tab2)
        image_label1_tab2.image = photo1_tab2

def proceed2():
    global loaded_image_tab2
    if loaded_image_tab2:
        resized_image = loaded_image_tab2.resize((250, 250))
        photo2_tab2 = ImageTk.PhotoImage(resized_image)
        image_label2_tab2.config(image=photo2_tab2)
        image_label2_tab2.image = photo2_tab2

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

empty_image = ImageTk.PhotoImage(Image.new('RGB', (250, 250), (200, 200, 200)))

image_label1_tab1 = tk.Label(tab1, image=empty_image)
image_label1_tab1.grid(row=1, column=0, padx=10, pady=10)

image_label2_tab1 = tk.Label(tab1, image=empty_image)
image_label2_tab1.grid(row=1, column=1, padx=10, pady=10)

tab2 = ttk.Frame(notebook)
notebook.add(tab2, text='DICOM image prediction')

upload_button_tab2 = tk.Button(tab2, text="Upload Image", command=upload_image_tab2)
upload_button_tab2.grid(row=0, column=0, padx=10, pady=10)

proceed_button_tab2 = tk.Button(tab2, text="Predict", command=proceed2)
proceed_button_tab2.grid(row=0, column=1, padx=10, pady=10)

image_label1_tab2 = tk.Label(tab2, image=empty_image)
image_label1_tab2.grid(row=1, column=0, padx=10, pady=10)

image_label2_tab2 = tk.Label(tab2, image=empty_image)
image_label2_tab2.grid(row=1, column=1, padx=10, pady=10)

root.mainloop()
