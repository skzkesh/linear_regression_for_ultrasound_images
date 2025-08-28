import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import numpy as np
 
# Define dataset
class RotationDataset(Dataset):
    def __init__(self, dataFrame, transform=None):
        self.data = dataFrame
        self.transform = transform
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['filename']
        angle = self.data.iloc[idx]['angle']
 
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
 
        # Scale angle to [-1, 1] just like training
        angle = angle / 45.0
        return image, torch.tensor(angle, dtype=torch.float32)
 
# Accuracy function
def within_margin(pred_scaled, tgt_scaled, margin=25.0):
    pred_deg = pred_scaled * 45.0
    tgt_deg = tgt_scaled * 45.0
    diff = np.abs(pred_deg - tgt_deg)
    return np.mean(diff <= margin) * 100
 
# Load test data
test_df = pd.read_csv('your_test_data_file.csv')
test_df['filename'] = test_df['filename'].apply(lambda x: "data/test/" + x.replace("\\", "/"))

# Dataset transformation
imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    imagenet_norm
])
 
test_dataset = RotationDataset(test_df, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
 
# Assign device to run the execution
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the existing weight
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load('resnet18_carotid_angle.pth', map_location=device))
model = model.to(device)
model.eval()
 
# Testing
y_true, y_pred = [], []
 
with torch.no_grad():
    for images, angles in test_loader:
        images = images.to(device)
        angles = angles.to(device).unsqueeze(1)
 
        outputs = model(images)
 
        y_true.extend(angles.cpu().numpy().flatten())
        y_pred.extend(outputs.cpu().numpy().flatten())
 
y_true = np.array(y_true)
y_pred = np.array(y_pred)
 
# Compute ±25° accuracy
accuracy = within_margin(y_pred, y_true, margin=25.0)
print(f"Test Accuracy within ±25°: {accuracy:.2f}%")
 
# Scatter plot: True vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_true*45, y_pred*45, alpha=0.5)
plt.plot([y_true.min()*45, y_true.max()*45],
         [y_true.min()*45, y_true.max()*45],
         'r--', label="Ideal")
plt.xlabel("True Angle")
plt.ylabel("Predicted Angle")
plt.title("True vs Predicted Angles (±25° tolerance)")
plt.legend()
plt.savefig('testing_scatter.png')
plt.show()
