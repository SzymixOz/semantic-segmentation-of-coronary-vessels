import torch
from torchvision import transforms
import cv2
import numpy as np
from .model import UNet


SIZE = 256

def predict(dicom, binary, keypoints, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet()
    model = model.to(device)

    model.load_state_dict(torch.load(f'semantic_segmentation/{model_name}.pth', map_location=device))
    model.eval()

    dicom = cv2.cvtColor(dicom, cv2.COLOR_RGBA2RGB)
    dicom = cv2.resize(dicom, (SIZE, SIZE))
    binary = cv2.resize(binary, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    keypoints = cv2.resize(keypoints, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

    binary = binary * 28
    binary = np.expand_dims(binary, axis=2)
    keypoints = np.expand_dims(keypoints, axis=2)
    input = np.concatenate((dicom, binary, keypoints), axis=2)

    input = transforms.ToTensor()(input).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model.predict(input)
        prediction = prediction.squeeze().cpu().numpy()
    
    return prediction
        
