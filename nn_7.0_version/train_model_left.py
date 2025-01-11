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
    def __init__(self, dicom_dir, binary_dir, keypoint_dir, output_dir, transform=None):
        self.dicom_dir = dicom_dir
        self.binary_dir = binary_dir
        self.keypoint_dir = keypoint_dir
        self.output_dir = output_dir
        self.transform = transform
        self.input_files = os.listdir(binary_dir)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        SIZE = 256
        dicom_path = os.path.join(self.dicom_dir, self.input_files[idx])
        binary_path = os.path.join(self.binary_dir, self.input_files[idx])
        keypoint_path = os.path.join(self.keypoint_dir, self.input_files[idx])
        output_path = os.path.join(self.output_dir, self.input_files[idx])

        dicom = cv2.imread(dicom_path, cv2.IMREAD_UNCHANGED)
        dicom = cv2.cvtColor(dicom, cv2.COLOR_RGBA2RGB)
        dicom = cv2.resize(dicom, (SIZE, SIZE))

        binary = cv2.imread(binary_path, cv2.IMREAD_UNCHANGED)
        binary = cv2.resize(binary, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

        keypoint = cv2.imread(keypoint_path, cv2.IMREAD_UNCHANGED)
        keypoint = cv2.resize(keypoint, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

        binary[binary > 0] += 28
        binary = np.expand_dims(binary, axis=2)
        keypoint[keypoint > 0] += 30
        keypoint = np.expand_dims(keypoint, axis=2)
        input = np.concatenate((dicom, binary, keypoint), axis=2)

        output = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
        output = cv2.resize(output, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        
        if self.transform:
            input = self.transform(input)
            output = torch.tensor(output, dtype=torch.long)
        
        return input, output
        

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dicom_dir = '../images/images_left/images_train/input_dicom'
train_binary_dir = '../images/images_left/images_train/binary'
train_keypoint_dir = '../images/images_left/images_train/keypoints'
train_output_dir = '../images/images_left/images_train/output'

val_dicom_dir = '../images/images_left/images_val/input_dicom'
val_binary_dir = '../images/images_left/images_val/binary'
val_keypoint_dir = '../images/images_left/images_val/keypoints'
val_output_dir = '../images/images_left/images_val/output'

train_dataset = CoronarySmallDataset(train_dicom_dir, train_binary_dir, train_keypoint_dir, train_output_dir, transform=transform)
val_dataset = CoronarySmallDataset(val_dicom_dir, val_binary_dir, val_keypoint_dir, val_output_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


train_losses = []
train_acc = []
val_losses = []
val_acc = []

def train_model(model, train_loader, val_loader, criterion, optimizer, model_name, num_epochs=50, early_stopping=3, lr_epoch_change=30, new_lr=0.001):
    best_loss = float('inf')
    best_val_accuracy = 0.0
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
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), f"{model_name}.pth")
        
        if epoch - epoch_with_improvement > early_stopping:
            print('Early stopping')
            break

        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(dropout_rate=0.4)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1.6e-4)

train_model(model, train_loader, val_loader, criterion, optimizer,
            "model_left_2", num_epochs=160, early_stopping=4)


starting_epoch = 3
y = list(range(starting_epoch, len(train_losses)))
plt.figure(figsize=(20, 10))
plt.plot(y, train_losses[starting_epoch:], label='Train Loss')
plt.scatter(y, train_losses[starting_epoch:])
plt.plot(y, val_losses[starting_epoch:], label='Validation Loss')
plt.scatter(y, val_losses[starting_epoch:])

for i in range(0, len(y), 20):
    plt.text(y[i], train_losses[starting_epoch + i] + 0.04, 
             f'{train_losses[starting_epoch + i]:.3f}', fontsize=12, ha='center')

for i in range(0, len(y), 20):
    plt.text(y[i], val_losses[starting_epoch + i] - 0.08, 
             f'{val_losses[starting_epoch + i]:.3f}', fontsize=12, ha='center')
    
for i in range(0, len(y), 20):
    plt.text(y[i], train_acc[starting_epoch + i] + 0.06, 
             f'{train_acc[starting_epoch + i] * 100:.2f}', fontsize=12, color='m', ha='center')
plt.text(y[len(y) - 1], train_acc[len(y) - 1] + 0.06, 
         f'{train_acc[len(y) - 1] * 100:.2f}', fontsize=12, color='m', ha='center')

for i in range(0, len(y), 20):
    plt.text(y[i], val_acc[starting_epoch + i] - 0.12, 
             f'{val_acc[starting_epoch + i] * 100:.2f}', fontsize=12, color='m', ha='center')
plt.text(y[len(y) - 1], val_acc[len(y) - 1] - 0.12, 
         f'{val_acc[len(y) - 1] * 100:.2f}', fontsize=12, color='m', ha='center')
    
plt.plot(y, train_acc[starting_epoch:], label='Train Acc')
plt.scatter(y, train_acc[starting_epoch:])
plt.plot(y, val_acc[starting_epoch:], label='Validation Acc')
plt.scatter(y, val_acc[starting_epoch:])

plt.xlabel('Epoch')
plt.ylabel('Loss, Acc')
plt.legend()
plt.title('Training and Validation Loss, Acc')
plt.savefig('loss_plot_left_2.png')
plt.show()