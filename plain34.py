!pip install gdown -q

!gdown --id 14Zt1EXmBWimBKBMOjQuwrlBY475d0Yze -O /content/IF25-4041-dataset.zip -q
!unzip /content/IF25-4041-dataset.zip -d /content/dataset

# Instalasi Dependensi
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install timm albumentations torchinfo scikit-learn pandas numpy matplotlib seaborn

# Import Library dan Konfigurasi
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
from torchinfo import summary

# Albumentations untuk augmentasi
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Scikit-learn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Set seed untuk reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Cek ketersediaan GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device yang digunakan: {device}")

# Definisi kelas
classes = ['bakso', 'gado_gado', 'nasi_goreng', 'rendang', 'soto_ayam']
num_classes = len(classes)
print(f"Jumlah kelas: {num_classes}")
print(f"Kelas: {classes}")

# Dataset Custom Class
class FoodDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_test=False):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

        if not is_test:
            # Mapping label ke index untuk training
            self.label_to_idx = {label: idx for idx, label in enumerate(classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, img_name)

        # Load image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            if isinstance(self.transform, A.Compose):
                # Albumentations transform
                image = np.array(image)
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                # Torchvision transform
                image = self.transform(image)

        if self.is_test:
            return image, img_name
        else:
            label = self.data.iloc[idx]['label']
            label_idx = self.label_to_idx[label]
            return image, label_idx

# Transformasi hanya resize dan normalisasi
train_transform = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Transformasi untuk validation/test tanpa augmentasi
val_transform = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

print("Dataset dan transformasi telah didefinisikan!")

"""
**Arsitektur Plain-34:**
- Initial conv layer (7x7, stride=2)
- MaxPool (3x3, stride=2)
- 4 stages dari Plain blocks:
  - Stage 1: 3 blocks, 64 channels
  - Stage 2: 4 blocks, 128 channels, stride=2 untuk first block
  - Stage 3: 6 blocks, 256 channels, stride=2 untuk first block
  - Stage 4: 3 blocks, 512 channels, stride=2 untuk first block
- Global Average Pool
- Fully Connected layer untuk 5 kelas makanan Indonesia

**Modifikasi:**
- Output classes: 5 kelas makanan Indonesia (bakso, gado_gado, nasi_goreng, rendang, soto_ayam)
- Training dari scratch tanpa pre-trained weights
"""

# Plain-34 Model Implementation
# Plain-34 Network: ResNet-34 architecture without residual connections
class PlainBlock(nn.Module):
    """
    Plain Block without residual connection.
    This is equivalent to a ResNet BasicBlock but without the skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(PlainBlock, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Downsample layer for dimension matching (if needed)
        self.downsample = downsample

    def forward(self, x):
        # Store input for potential downsampling
        identity = x

        # First conv + bn + relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second conv + bn
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply downsample to identity if needed (for dimension matching)
        if self.downsample is not None:
            identity = self.downsample(identity)

        # NO RESIDUAL CONNECTION HERE (this is the key difference from ResNet)
        # In ResNet, we would do: out += identity
        # But in Plain network, we just apply ReLU directly

        out = F.relu(out)

        return out

class Plain34(nn.Module):
    """
    Plain-34 Network: ResNet-34 architecture without residual connections.

    Architecture:
    - Initial conv layer (7x7, stride=2)
    - MaxPool (3x3, stride=2)
    - 4 stages of Plain blocks:
      - Stage 1: 3 blocks, 64 channels
      - Stage 2: 4 blocks, 128 channels, stride=2 for first block
      - Stage 3: 6 blocks, 256 channels, stride=2 for first block
      - Stage 4: 3 blocks, 512 channels, stride=2 for first block
    - Global Average Pool
    - Fully Connected layer
    """

    def __init__(self, num_classes=5):
        super(Plain34, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Plain block stages
        self.stage1 = self._make_stage(64, 64, 3, stride=1)    # 3 blocks, 64 channels
        self.stage2 = self._make_stage(64, 128, 4, stride=2)   # 4 blocks, 128 channels
        self.stage3 = self._make_stage(128, 256, 6, stride=2)  # 6 blocks, 256 channels
        self.stage4 = self._make_stage(256, 512, 3, stride=2)  # 3 blocks, 512 channels

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        """
        Create a stage consisting of multiple PlainBlocks.

        Args:
            in_channels: Input channels for the first block
            out_channels: Output channels for all blocks in this stage
            num_blocks: Number of blocks in this stage
            stride: Stride for the first block (usually 1 or 2)
        """
        downsample = None

        # If we need to change dimensions or stride, create downsample layer
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []

        # First block (may have stride=2 and different input/output channels)
        layers.append(PlainBlock(in_channels, out_channels, stride, downsample))

        # Remaining blocks (stride=1, same input/output channels)
        for _ in range(1, num_blocks):
            layers.append(PlainBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial conv + bn + relu + maxpool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        # Plain block stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Final classification layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def create_plain34(num_classes=5):
    """
    Factory function to create Plain-34 model.

    Args:
        num_classes: Number of output classes (default: 5 for Indonesian food dataset)

    Returns:
        Plain34 model instance
    """
    return Plain34(num_classes=num_classes)

def test_model():
    """
    Test function to verify the model works correctly.
    This function creates a model and prints its architecture summary.
    """
    print("Creating Plain-34 model...")
    model = create_plain34(num_classes=5)

    # Print model summary
    print("\n" + "="*50)
    print("PLAIN-34 MODEL ARCHITECTURE SUMMARY")
    print("="*50)

    # Test with typical input size for image classification (224x224)
    try:
        summary(model, input_size=(1, 3, 224, 224), verbose=1)
    except Exception as e:
        print(f"Error in torchinfo summary: {e}")
        print("Trying manual forward pass...")

        # Manual test
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            output = model(test_input)
            print(f"Input shape: {test_input.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Expected output shape: (1, 5)")
            print(f"Model works correctly: {output.shape == (1, 5)}")

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model

print("✅ Plain-34 Model berhasil didefinisikan!")
print("📝 Model ini merupakan ResNet-34 tanpa residual connections")

# Inisialisasi Model Plain-34 (menggunakan kode yang sudah didefinisikan di atas)
print("🔧 Membuat model Plain34...")
model = create_plain34(num_classes=5)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

print("=" * 60)
print("SUMMARY MODEL PLAIN-34")
print("=" * 60)
summary(model, input_size=(1, 3, 224, 224))

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n📊 STATISTIK MODEL:")
print(f"Total Parameter: {total_params:,}")
print(f"Trainable Parameter: {trainable_params:,}")
print(f"Parameter < 26M: {'✅ YA' if total_params < 26_000_000 else '❌ TIDAK'}")
print(f"Ukuran Model: ~{total_params * 4 / (1024**2):.1f} MB (float32)")

print(f"\n🧪 TEST FORWARD PASS:")
with torch.no_grad():
    test_input = torch.randn(1, 3, 224, 224).to(device)
    test_output = model(test_input)
    print(f"✓ Input shape: {test_input.shape}")
    print(f"✓ Output shape: {test_output.shape}")
    print(f"✓ Output classes: {test_output.argmax(dim=1).item()} (predicted class)")
    print(f"✓ Forward pass berhasil!")
print("✅ Model Plain-34 siap digunakan!")

# Setup training components
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Contoh data untuk testing
images = torch.randn(16, 3, 224, 224).to(device)
labels = torch.randint(0, 5, (16,)).to(device)

# Test training step
model.train()
optimizer.zero_grad()

outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()

print("\n---")
print("✅ Model Plain34 berhasil diintegrasikan dan siap untuk training.")
print(f"Loss pada langkah test ini: {loss.item():.4f}")
print("🔍 Model menggunakan arsitektur ResNet-34 TANPA residual connections")

"""## Arsitektur Model Plain-34

**Mengapa Plain-34?**
- Plain-34 adalah versi ResNet-34 **tanpa residual connections (skip connections)**
- Berguna untuk memahami dampak residual connections dalam deep learning
- Training dari scratch untuk melihat kemampuan model tanpa transfer learning
- Parameter count yang masih dalam batas (~21M, masih dalam batas < 26M)
- Implementasi yang clean dan mudah dipahami

**Detail Arsitektur:**
- **Backbone**: Plain blocks tanpa skip connections
- **Training**: Dari scratch tanpa pre-trained weights
- **Output Classes**: 5 kelas makanan Indonesia
- **Initialization**: He initialization untuk optimal training

**Plain Block vs ResNet Block:**
- Plain Block: conv -> bn -> relu -> conv -> bn -> relu
- ResNet Block: conv -> bn -> relu -> conv -> bn -> (ADD skip connection) -> relu
- **Perbedaan utama**: Plain-34 tidak memiliki skip connection sehingga lebih sulit untuk dilatih pada network yang dalam

## Training Setup

**Hiperparameter:**
- Learning Rate: 1e-4 dengan scheduler
- Batch Size: 32
- Epochs: 50 (dengan early stopping)
- Optimizer: AdamW
- Loss Function: CrossEntropyLoss
- Scheduler: CosineAnnealingLR

**Catatan Penting:**
- **Tidak menggunakan data augmentation** sesuai permintaan
- **Tidak melakukan analisis data** sesuai permintaan
- Model Plain-34 training dari scratch (tanpa pre-trained weights)
- **Tracking metrik kunci**: Training accuracy, Validation accuracy, Training loss, Validation loss
- Model akan dilatih dan bobot disimpan sebagai checkpoint untuk evaluasi
"""

# TRAINING SETUP DAN EKSEKUSI
# Definisi hyperparameter
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 5

# Load dan split training data
print("🔄 Mempersiapkan data training...")
train_df = pd.read_csv('dataset/train.csv')
print(f"✓ Dataset dimuat: {len(train_df)} sampel")

# Split training data dengan stratified sampling
train_data, val_data = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=42)
print(f"✓ Data split - Train: {len(train_data)}, Val: {len(val_data)}")

# Simpan split data untuk reproducibility
train_data.to_csv('train_split.csv', index=False)
val_data.to_csv('val_split.csv', index=False)
print("✓ Split data disimpan ke train_split.csv dan val_split.csv")

# Dataset dan DataLoader
print("🔄 Membuat DataLoader...")
train_dataset = FoodDataset('train_split.csv', 'dataset/train', transform=train_transform)
val_dataset = FoodDataset('val_split.csv', 'dataset/train', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)  # num_workers=0 untuk Windows
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"✓ DataLoader siap - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# Loss function dan optimizer
print("🔄 Menyiapkan loss function dan optimizer...")
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

if EPOCHS > 5:
    base_cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS-5)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[LinearLR(optimizer, start_factor=0.1, total_iters=5), base_cosine],
        milestones=[5],
    )
else:
    # fallback kalau epoch sedikit
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
print("✓ Loss function dan optimizer siap")

def save_ckpt(path, epoch, best_acc, train_losses, val_losses, train_accs, val_accs):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_acc': best_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }, path)

# Training loop
def train_model():
    print("🚀 Memulai training Plain-34...")
    print("📊 Melacak metrik kunci: Training Acc, Validation Acc, Training Loss, Validation Loss")

    best_acc = 0.0
    patience = 10
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print("\n" + "="*80)
    print(f"{'EPOCH':<6} {'LR':<10} {'TRAIN_LOSS':<12} {'TRAIN_ACC':<12} {'VAL_LOSS':<12} {'VAL_ACC':<12}")
    print("="*80)

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.detach(), 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.detach(), 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        # Calculate metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Display metrics
        current_lr = optimizer.param_groups[0]['lr']
        print(f"{epoch+1:<6} {current_lr:<10.2e} {avg_train_loss:<12.4f} {train_acc:<12.4f} {avg_val_loss:<12.4f} {val_acc:<12.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_ckpt('model_best.pth', epoch+1, best_acc, train_losses, val_losses, train_accs, val_accs)
            print(f"    ✅ Best checkpoint saved! Val Acc: {val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    🛑 Early stopping triggered at epoch {epoch+1}")
                break

        scheduler.step()

    print("="*80)
    print(f"🎉 Training selesai!")
    print(f"📊 Final Results:")
    print(f"   Best Validation Accuracy: {best_acc:.4f}")
    print(f"   Final Training Loss: {train_losses[-1]:.4f}")
    print(f"   Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"   Final Training Accuracy: {train_accs[-1]:.4f}")

    return train_losses, val_losses, train_accs, val_accs

# Jalankan training
print("▶️  Memulai training Plain-34...")
train_losses, val_losses, train_accs, val_accs = train_model()

print("\n✅ Training telah selesai! Metrics telah dicatat dan disimpan.")

# Plot Training Metrics
print("📈 Membuat visualisasi metrik training...")

# Plot metrics
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

epochs_range = range(1, len(train_losses) + 1)

# Plot 1: Training vs Validation Loss
ax1.plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2)
ax1.plot(epochs_range, val_losses, 'r-', label='Validation Loss', linewidth=2)
ax1.set_title('Training dan Validation Loss', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Training vs Validation Accuracy
ax2.plot(epochs_range, train_accs, 'b-', label='Training Accuracy', linewidth=2)
ax2.plot(epochs_range, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
ax2.set_title('Training dan Validation Accuracy', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Loss Difference (Overfitting indicator)
loss_diff = [val - train for val, train in zip(val_losses, train_losses)]
ax3.plot(epochs_range, loss_diff, 'g-', linewidth=2)
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax3.set_title('Loss Difference (Val - Train)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss Difference')
ax3.grid(True, alpha=0.3)
ax3.fill_between(epochs_range, loss_diff, alpha=0.3, color='green' if loss_diff[-1] < 0.5 else 'orange')

# Plot 4: Accuracy Difference (Generalization gap)
acc_diff = [train - val for val, train in zip(val_accs, train_accs)]
ax4.plot(epochs_range, acc_diff, 'purple', linewidth=2)
ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax4.set_title('Accuracy Difference (Train - Val)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Accuracy Difference')
ax4.grid(True, alpha=0.3)
ax4.fill_between(epochs_range, acc_diff, alpha=0.3, color='green' if acc_diff[-1] < 0.1 else 'orange')

plt.tight_layout()
plt.show()

# Print detailed metrics summary
print("\n" + "="*60)
print("📊 RINGKASAN METRIK TRAINING PLAIN-34")
print("="*60)

best_train_acc = max(train_accs)
best_val_acc = max(val_accs)
final_train_acc = train_accs[-1]
final_val_acc = val_accs[-1]
min_train_loss = min(train_losses)
min_val_loss = min(val_losses)
final_train_loss = train_losses[-1]
final_val_loss = val_losses[-1]

print(f"🏆 BEST METRICS:")
print(f"   Best Training Accuracy: {best_train_acc:.4f}")
print(f"   Best Validation Accuracy: {best_val_acc:.4f}")
print(f"   Minimum Training Loss: {min_train_loss:.4f}")
print(f"   Minimum Validation Loss: {min_val_loss:.4f}")

print(f"\n📈 FINAL METRICS:")
print(f"   Final Training Accuracy: {final_train_acc:.4f}")
print(f"   Final Validation Accuracy: {final_val_acc:.4f}")
print(f"   Final Training Loss: {final_train_loss:.4f}")
print(f"   Final Validation Loss: {final_val_loss:.4f}")

print(f"\n🔍 ANALYSIS:")
overfitting_indicator = final_val_loss - final_train_loss
generalization_gap = final_train_acc - final_val_acc

print(f"   Overfitting Indicator (Val Loss - Train Loss): {overfitting_indicator:.4f}")
print(f"   Generalization Gap (Train Acc - Val Acc): {generalization_gap:.4f}")

if overfitting_indicator > 0.5:
    print("   ⚠️  Model mungkin mengalami overfitting (loss difference > 0.5)")
else:
    print("   ✅ Model menunjukkan generalisasi yang baik")

if generalization_gap > 0.15:
    print("   ⚠️  Generalization gap cukup besar (> 0.15)")
else:
    print("   ✅ Generalization gap dalam batas wajar")

print("="*60)

# Fungsi Utilitas Training
def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """
    Plot grafik training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_title('Training dan Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(train_accs, label='Train Acc', color='blue')
    ax2.plot(val_accs, label='Val Acc', color='red')
    ax2.set_title('Training dan Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def evaluate_model(model, data_loader, dataset_name="Test"):
    """
    Evaluasi model pada dataset
    """
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.detach(), 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f"{dataset_name} Accuracy: {accuracy:.4f}")

    # Classification report
    class_names = classes
    print(f"\n{dataset_name} Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))

    return accuracy, all_predictions, all_labels

def test_single_batch():
    """
    Test single batch untuk memastikan DataLoader bekerja
    """
    print("🔍 Testing DataLoader dengan single batch...")
    try:
        # Test training loader
        train_iter = iter(train_loader)
        images, labels = next(train_iter)
        print(f"✓ Train batch shape: {images.shape}, Labels shape: {labels.shape}")
        print(f"✓ Sample labels: {labels[:5].tolist()}")
        print(f"✓ Label classes: {[classes[i] for i in labels[:5]]}")

        # Test validation loader
        val_iter = iter(val_loader)
        images, labels = next(val_iter)
        print(f"✓ Val batch shape: {images.shape}, Labels shape: {labels.shape}")

        # Test model forward pass
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            print(f"✓ Model output shape: {outputs.shape}")
            print(f"✓ Sample predictions: {torch.argmax(outputs, dim=1)[:5].tolist()}")

        print("✅ DataLoader dan model test berhasil!")
        return True

    except Exception as e:
        print(f"❌ Error dalam test: {str(e)}")
        return False

# Jalankan test
test_result = test_single_batch()