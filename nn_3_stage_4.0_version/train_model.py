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
import matplotlib.pyplot as plt


class CoronarySmallDataset(Dataset):
    def __init__(self, image_dir, keypoint_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.keypoint_dir = keypoint_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.input = os.listdir(image_dir)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        SIZE = 256
        img_path = os.path.join(self.image_dir, self.input[idx])
        keypoint_path = os.path.join(self.keypoint_dir, self.input[idx])
        mask_path = os.path.join(self.mask_dir, self.input[idx])
        

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image = cv2.resize(image, (SIZE, SIZE))

        keypoint = cv2.imread(keypoint_path, cv2.IMREAD_UNCHANGED)
        keypoint = cv2.resize(keypoint, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        
        keypoint =  np.expand_dims(keypoint, axis=2)
        output = np.concatenate((image, keypoint), axis=2)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        
        if self.transform:
            output = self.transform(output)
            mask = torch.tensor(mask, dtype=torch.long)
        
        return output, mask
        

transform = transforms.Compose([
    transforms.ToTensor()
])

train_image_dir = '../images/images_train_original/input_dicom'
train_keypoint_dir = '../images/images_train_original/keypoints'
train_mask_dir = '../images/images_train_original/output'
val_image_dir = '../images/images_val/input_dicom'
val_keypoint_dir = '../images/images_val/keypoints'
val_mask_dir = '../images/images_val/output'

train_dataset = CoronarySmallDataset(train_image_dir, train_keypoint_dir, train_mask_dir, transform=transform)
val_dataset = CoronarySmallDataset(val_image_dir, val_keypoint_dir, val_mask_dir, transform=transform)
# test_dataset = CoronarySmallDataset(test_image_dir, test_keypoint_dir, test_mask_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


train_losses = []
train_acc = []
val_losses = []
val_acc = []

def train_model(model, train_loader, val_loader, criterion, optimizer, model_name, num_epochs=50, early_stopping=3, lr_epoch_change=30, new_lr=0.001):
    best_loss = float('inf')
    best_acc = 0.0
    epoch_with_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_start_time = time.time()
        
        for inputs, outputs in tqdm(train_loader):
            inputs = inputs.to(device)
            outputs = outputs.to(device)

            optimizer.zero_grad()
            predictions = model(inputs)
            
            loss = criterion(predictions, outputs)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            # Obliczanie dokładności
            _, predicted = torch.max(predictions, 1)
            # train_total += outputs.numel()
            train_total += (outputs != 0).sum().item()
            train_correct += ((predicted == outputs) & (outputs != 0)).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracy = train_correct / train_total
        train_acc.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, outputs in val_loader:
                inputs = inputs.to(device)
                outputs = outputs.to(device)

                predictions = model(inputs)
                loss = criterion(predictions, outputs)
                
                val_loss += loss.item() * inputs.size(0)
                
                # Obliczanie dokładności
                _, predicted = torch.max(predictions, 1)
                # val_total += outputs.numel()
                val_total += (outputs != 0).sum().item()
                val_correct += ((predicted == outputs) & (outputs != 0)).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracy = val_correct / val_total
        val_acc.append(val_accuracy)
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy * 100:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy * 100:.2f}%, "
              f"Time: {(epoch_time // 60):.0f}min {(epoch_time % 60):.1f}s")
        
        if val_loss < best_loss:
            best_loss = val_loss
            epoch_with_improvement = epoch
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            epoch_with_improvement = epoch
            torch.save(model.state_dict(), f"{model_name}.pth")
        
        if epoch - epoch_with_improvement > early_stopping:
            print('Early stopping')
            break

        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(dropout_rate=0.4)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2.0e-4)

train_model(model, train_loader, val_loader, criterion, optimizer,
            "model_dicom_kp_2", num_epochs=200, early_stopping=5)


starting_epoch = 3
y = list(range(starting_epoch, len(train_losses)))
plt.figure(figsize=(20, 10))
plt.plot(y, train_losses[starting_epoch:], label='Train Loss')
plt.scatter(y, train_losses[starting_epoch:])
plt.plot(y, val_losses[starting_epoch:], label='Validation Loss')
plt.scatter(y, val_losses[starting_epoch:])

# Dodawanie wartości co 10 punktów dla Train Loss powyżej punktów
for i in range(0, len(y), 10):
    plt.text(y[i], train_losses[starting_epoch + i] + 0.03, 
             f'{train_losses[starting_epoch + i]:.3f}', fontsize=12, ha='center')

# Dodawanie wartości co 10 punktów dla Validation Loss poniżej punktów
for i in range(0, len(y), 10):
    plt.text(y[i], val_losses[starting_epoch + i] - 0.07, 
             f'{val_losses[starting_epoch + i]:.3f}', fontsize=12, ha='center')
    
plt.plot(y, train_acc[starting_epoch:], label='Train Acc')
plt.scatter(y, train_acc[starting_epoch:])
plt.plot(y, val_acc[starting_epoch:], label='Validation Acc')
plt.scatter(y, val_acc[starting_epoch:])

plt.xlabel('Epoch')
plt.ylabel('Loss, Acc')
plt.legend()
plt.title('Training and Validation Loss, Acc')
plt.savefig('loss_plot_dicom_kp_2.png')
plt.show()