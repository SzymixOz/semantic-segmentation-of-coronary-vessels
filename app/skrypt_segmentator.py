import cv2
import torch
import torchvision
from torchvision import transforms
import numpy as np
import torch.nn as nn
import sys
from binary_segmentation.skrypt import process_dicom_image
from keypoints.keypoints import get_keypoints
from semantic_segmentation.predict import predict
from semantic_segmentation.converter_RGB import get_colored_mask_with_legend
from save_result import save_result_as_csv

SIZE = 256

def load_side_model(device):
    """Load and prepare the side recognition model."""
    pretrained_net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    side_model = nn.Sequential(
        pretrained_net,
        nn.ReLU(),
        nn.Linear(1000, 2)
    ).to(device)
    side_model.load_state_dict(torch.load('side_recognition_model.pth', map_location=device))
    side_model.eval()
    return side_model

def perform_side_recognition(binary_mask, device):
    """Perform side recognition on the given binary mask."""
    binary_mask = cv2.resize(binary_mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
    side_model = load_side_model(device)

    img_tensor = transforms.ToTensor()(binary_mask).unsqueeze(0).to(device)
    with torch.no_grad():
        side_pred = side_model(img_tensor)
        recognized_side = torch.argmax(side_pred).item() == 1

    detected_side_str = "right" if recognized_side else "left"
    print(f"Detected side: {detected_side_str}")
    return recognized_side

def run_pipeline(image_path):
    """Run the semantic segmentation pipeline."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded_image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(loaded_image, cv2.COLOR_BGRA2RGB)
    if loaded_image is None:
        print("Error: Unable to load image.")
        return

    print("Performing binary segmentation...")
    binary_mask = process_dicom_image(image_rgb)

    print("Performing side recognition...")
    recognized_side = perform_side_recognition(binary_mask, device)

    print("Performing semantic segmentation...")
    keypoints = get_keypoints(binary_mask)
    model_name = 'model_right' if recognized_side else 'model_left'
    segmentation_result = predict(loaded_image, binary_mask, keypoints, model_name)
  
    # Generate result with legend
    result_with_legend = get_colored_mask_with_legend(segmentation_result)

    # Save results
    save_result_as_csv(segmentation_result, "results/segmentation_mask.csv")
    output_path = "results/segmentation_result.png"
    cv2.imwrite(output_path, result_with_legend)
    print(f"Segmentation result saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skrypt_segmentator.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    run_pipeline(image_path)
