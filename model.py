"""
Model Documentation

Architecture:
- The model uses a modified 3D ResNet architecture (r3d_18) for binary classification of MRI scans.
- The input layer is adjusted to accept single-channel images.

Training Process:
- The model is trained using the Adam optimizer and CrossEntropyLoss.
- Training occurs over 10 epochs, with metrics logged for each epoch.

Hyperparameters:
- Learning Rate: 0.001
- Batch Size: 4
- Number of Epochs: 10

Dataset:
- The dataset consists of MRI images loaded from the specified directory.
- Labels indicate the presence of a tumor (1 for tumor, 0 for no tumor).

Metrics:
- The model tracks training loss, validation loss, and validation accuracy.

"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Normalize
import torch.nn as nn
import torch.optim as optim
import nibabel as nib
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import logging
from torch.nn.utils import prune
from torchvision.models.video import r3d_18

# Dataset preparation
class MRIDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# MRI Data Handling
class MRI:
    def __init__(self, data_dir='data/BraTS2021_Training_Data', normalize=False):
        self.images = []
        self.labels = []
        self.load_data(data_dir)
        if normalize:
            self.images = self.normalize_images(self.images)

    def load_data(self, data_dir):
        for patient_folder in os.listdir(data_dir):
            patient_path = os.path.join(data_dir, patient_folder)
            if os.path.isdir(patient_path):
                flair_path = os.path.join(patient_path, f"{patient_folder}_flair.nii.gz")
                seg_path = os.path.join(patient_path, f"{patient_folder}_seg.nii.gz")
                # Load NIfTI files
                if os.path.exists(flair_path) and os.path.exists(seg_path):
                    flair_img = nib.load(flair_path).get_fdata()
                    self.images.append(flair_img)
                    self.labels.append(1 if 'tumor' in patient_folder else 0)

        # Check if images and labels are populated
        if not self.images or not self.labels:
            raise ValueError("No images or labels found. Please check the data directory.")

    def normalize_images(self, images):
        return [(img - np.min(img)) / (np.max(img) - np.min(img)) for img in images]

# Resize function for 3D images
def resize_3d_image(image, new_shape):
    zoom_factors = [new / old for new, old in zip(new_shape, image.shape)]
    return zoom(image, zoom_factors)

class Resize3D:
    def __init__(self, new_shape):
        self.new_shape = new_shape

    def __call__(self, image):
        return resize_3d_image(image, self.new_shape)

# Function to export the model to ONNX format
def export_model_to_onnx(model, model_path='best_model.pth', onnx_path='model.onnx'):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    dummy_input = torch.randn(1, 1, 224, 224, 224)  # Adjust input shape for 3D model
    torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=10, do_constant_folding=True, input_names=['input'], output_names=['output'])
    print(f'Model exported to {onnx_path}')

# Function to prune the model
def prune_model(model, amount=0.2):
    """
    Prune the model to reduce its size and improve inference speed.
    """
    for module in model.modules():
        if isinstance(module, nn.Conv3d):
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
            prune.remove(module, 'weight')  # Ensure the mask is removed after pruning

# Add channel dimension to images
def add_channel_dimension(image):
    return np.expand_dims(image, axis=0)

def to_tensor(image):
    return torch.tensor(image, dtype=torch.float32)

if __name__ == '__main__':
    # Initialize MRI data and transform
    mri_data = MRI(data_dir='data', normalize=True)
    transform = Compose([
        Resize3D((128, 128, 128)),  # Resize 3D images
        add_channel_dimension,  # Add channel dimension
        to_tensor,  # Convert to tensor
        Normalize(mean=[0.5], std=[0.5])  # Normalize
    ])

    # Dataset and DataLoader
    dataset = MRIDataset(mri_data.images, mri_data.labels, transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

    # Logging setup
    logging.basicConfig(level=logging.INFO, filename='training/training_log.txt', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

    # Load and modify the model
    model = r3d_18(weights='R3D_18_Weights.DEFAULT')
    model.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Adjust the model output to match the expected target shape
    model.fc = nn.Linear(model.fc.in_features, 2)  # Ensure the output layer matches the number of classes

    # Prune the model
    prune_model(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize lists to track metrics
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Training loop
    best_accuracy = 0.0
    for epoch in range(10):  # Changed to 10 epochs
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.view(-1, 2), labels)  # Adjust output shape for loss calculation
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs.view(-1, 2), labels)  # Adjust output shape for loss calculation
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accuracy = val_correct / val_total
        val_accuracies.append(val_accuracy)

        logging.info(f'Epoch [{epoch+1}/10], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            logging.info('Model saved!')

    # Export the model to ONNX format after training
    export_model_to_onnx(model)

    # Plot metrics
    epochs = range(1, 11)  # Adjusted for 10 epochs
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training/training_validation_plot.png')
    plt.show()
