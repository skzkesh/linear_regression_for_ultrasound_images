import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
 
# Define dataset format
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
 
        # Scale angle to [-1, 1]
        angle = angle / 45.0
        return image, torch.tensor(angle, dtype=torch.float32)
 
# Define accuracy function (allow +- 25 degree error)
def within_margin(pred_scaled, tgt_scaled, margin=25.0):
    # Rescale back to degrees
    pred_deg = pred_scaled * 45.0
    tgt_deg = tgt_scaled * 45.0
    diff = (pred_deg - tgt_deg).abs()
    return (diff <= margin).float().mean().item()
 
# Load dataset
train_df = pd.read_csv('your_csv_train_file.csv')
train_df['filename'] = train_df['filename'].apply(lambda x: "data/" + x.replace("\\", "/"))
 
val_df = pd.read_csv('your_csv_val_file.csv')
val_df['filename'] = val_df['filename'].apply(lambda x: "data/" + x.replace("\\", "/"))
 
# Data transformation and augmentation
# Ensure data transformation does not include rotate flip, or anything that effect the angle unless we update the angles accordingly
imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
 
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    imagenet_norm
])
 
aug_transform1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(saturation=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    imagenet_norm
])
 
aug_transform2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    imagenet_norm
])
 
aug_transform3 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    imagenet_norm
])

# Combine original data with augmented data
train_dataset = ConcatDataset([
    RotationDataset(train_df, transform=base_transform),
    RotationDataset(train_df, transform=aug_transform1),
    RotationDataset(train_df, transform=aug_transform2),
    RotationDataset(train_df, transform=aug_transform3)
])
 
val_dataset = RotationDataset(val_df, transform=base_transform)
 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
 
# Load our pre-trained model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)  # Regression output

# Assign device to run the execution 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 0 is the default GPU index
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(device))
 
model = model.to(device)

# Training setup
criterion = nn.SmoothL1Loss(beta=5.0)  # Huber loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)
 
num_epochs = 30
patience = 5
best_val_loss = float('inf')
early_stopping_counter = 0
 
train_losses, val_losses = [], []
 
# Train model
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
 
    for images, angles in train_loader:
        images = images.to(device)
        angles = angles.to(device).unsqueeze(1)  # shape: (batch_size, 1)
 
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, angles)
        loss.backward()
        optimizer.step()
 
        total_train_loss += loss.item()
 
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
 
    # Validation
    model.eval()
    total_val_loss, total_acc = 0.0, 0.0
    with torch.no_grad():
        for images, angles in val_loader:
            images = images.to(device)
            angles = angles.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, angles)
            total_val_loss += loss.item()
 
            # accuracy in degrees with ±25° margin
            acc = within_margin(outputs.detach().cpu(), angles.detach().cpu(), margin=25.0)
            total_acc += acc
 
    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_acc = total_acc / len(val_loader)
    val_losses.append(avg_val_loss)
 
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Acc (±25°): {avg_val_acc*100:.2f}%")
 
    scheduler.step(avg_val_loss)
 
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), 'resnet18_carotid_angle.pth')
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("Early stopping triggered.")
            break
 
# Plot losses
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('loss_curve.png')
plt.show()
