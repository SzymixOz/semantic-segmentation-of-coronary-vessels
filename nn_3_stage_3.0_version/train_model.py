import os
import time
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from large_RGB_model import UNet
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random


class CoronarySmallDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        SIZE = 256
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image = cv2.resize(image, (SIZE, SIZE))
        
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        # mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2RGB)
        mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        
        if self.transform:
            image = self.transform(image)
            # mask = self.transform(mask)
            mask = torch.tensor(mask, dtype=torch.long)
        
        return image, mask
        

transform = transforms.Compose([
    transforms.ToTensor()
])

train_image_dir = '../images/images_train/input_dicom'
train_mask_dir = '../images/images_train/output'
val_image_dir = '../images/images_val/input_dicom'
val_mask_dir = '../images/images_val/output'
# test_image_dir = '..\images\images_test\input'
# test_mask_dir = '..\images\images_test\output'

train_dataset = CoronarySmallDataset(train_image_dir, train_mask_dir, transform=transform)
val_dataset = CoronarySmallDataset(val_image_dir, val_mask_dir, transform=transform)
# test_dataset = CoronarySmallDataset(test_image_dir, test_mask_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet()
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1.2e-4)
train_losses = []
val_losses = []

def train_model(model, train_loader, val_loader, criterion, optimizer, model_name, num_epochs=50, early_stopping=3):
    best_loss = float('inf')
    epoch_with_improvement = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        epoch_start_time = time.time()
        for images, masks in tqdm(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item() * images.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f},  Time: {(epoch_time // 60):.0f}min {(epoch_time % 60):.1f}s' )
        
        if val_loss < best_loss:
            best_loss = val_loss
            epoch_with_improvement = epoch
            torch.save(model.state_dict(), f"{model_name}.pth")
        
        if epoch - epoch_with_improvement > early_stopping:
            print('Early stopping')
            break


train_model(model, train_loader, val_loader, criterion, optimizer,
            "model_dicom_1", num_epochs=100, early_stopping=5)


starting_epoch = 5
y = list(range(starting_epoch, len(train_losses)))
plt.figure(figsize=(100, 50))
plt.plot(y, train_losses[starting_epoch:], label='Train Loss')
plt.scatter(y, train_losses[starting_epoch:])
plt.plot(y, val_losses[starting_epoch:], label='Validation Loss')
plt.scatter(y, val_losses[starting_epoch:])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('loss_plot_dicom1.png')
plt.show()
