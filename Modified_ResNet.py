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

"""ResNet"""

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_dropout=False, dropout_p=0.5, downsample=None):
        super(ResNetBlock, self).__init__()

        # Dropout regularization setup
        # Dropout mencegah overfitting dengan secara acak mematikan neuron selama training
        # Ini memaksa model untuk tidak bergantung pada neuron tertentu dan meningkatkan generalization
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=dropout_p)

        # Pre-activation architecture: BatchNorm -> ReLU -> Conv
        # Pre-activation terbukti lebih efektif karena:
        # 1. Normalisasi input sebelum aktivasi memberikan gradient yang lebih stabil
        # 2. Aktivasi pada skip connection memberikan identitas yang bersih
        # 3. Mengurangi masalah gradient vanishing/exploding
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        # Pre-activation sequence: BN -> ReLU -> Conv
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        # Dropout regularization after first convolution
        # Mengurangi co-adaptation antar feature maps
        if self.use_dropout:
          out = self.dropout(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        return out


class MultiPathResNetBlock(nn.Module):
    """
    Multi-path ResNet Block dengan dua jalur konvolusi paralel (3x3 dan 5x5)
    
    Konsep Multi-path:
    - Menggabungkan feature dari kernel size berbeda (3x3 dan 5x5) secara paralel
    - 3x3 kernel: menangkap detail lokal dan fine-grained features
    - 5x5 kernel: menangkap konteks yang lebih luas dan spatial relationships
    - Kombinasi kedua jalur memberikan representasi yang lebih kaya
    - Mirip dengan konsep Inception module tapi lebih sederhana
    """
    def __init__(self, in_channels, out_channels, stride=1, use_dropout=False, dropout_p=0.5, downsample=None):
        super().__init__()

        # Dropout regularization untuk multi-path architecture
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=dropout_p)

        # Jalur 3x3: untuk detail lokal dan fine-grained features
        self.conv3a = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn3a = nn.BatchNorm2d(out_channels)
        self.conv3b = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn3b = nn.BatchNorm2d(out_channels)
        
        # Jalur 5x5: untuk konteks yang lebih luas dan spatial relationships
        self.conv5a = nn.Conv2d(in_channels, out_channels, 5, stride, 2, bias=False)
        self.bn5a = nn.BatchNorm2d(out_channels)
        self.conv5b = nn.Conv2d(out_channels, out_channels, 5, 1, 2, bias=False)
        self.bn5b = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        # Multi-path processing: jalur 3x3 dan 5x5 diproses secara paralel
        out3 = F.relu(self.bn3a(self.conv3a(x)))
        if self.use_dropout:
            out3 = self.dropout(out3)
        out3 = self.bn3b(self.conv3b(out3))
        
        out5 = F.relu(self.bn5a(self.conv5a(x)))
        if self.use_dropout:
            out5 = self.dropout(out5)
        out5 = self.bn5b(self.conv5b(out5))
        
        # Penggabungan multi-path: rata-rata dari kedua jalur
        # Memberikan bobot yang sama untuk kedua jenis feature
        out = (out3 + out5) / 2
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, num_classes=5, use_dropout=True, dropout_p=0.3):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self._make_stage(64, 64, 3, stride=1, use_dropout=use_dropout, dropout_p=dropout_p, multipath=True)
        self.stage2 = self._make_stage(64, 128, 4, stride=2, use_dropout=use_dropout, dropout_p=dropout_p, multipath=True)
        self.stage3 = self._make_stage(128, 256, 6, stride=2, use_dropout=use_dropout, dropout_p=dropout_p, multipath=True)
        self.stage4 = self._make_stage(256, 512, 3, stride=2, use_dropout=use_dropout, dropout_p=dropout_p, multipath=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout sebelum fully connected layer dengan probabilitas lebih tinggi
        # Dropout di akhir network lebih efektif untuk mencegah overfitting pada classifier
        self.final_dropout = nn.Dropout(p=dropout_p * 1.5) if use_dropout else nn.Identity()
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, num_blocks, stride, use_dropout=False, dropout_p=0.5, multipath=False):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        BlockClass = MultiPathResNetBlock if multipath else ResNetBlock
        layers.append(BlockClass(in_channels, out_channels, stride=stride, use_dropout=use_dropout, dropout_p=dropout_p, downsample=downsample))
        for _ in range(1, num_blocks):
            layers.append(BlockClass(out_channels, out_channels, use_dropout=use_dropout, dropout_p=dropout_p))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.final_dropout(x)
        x = self.fc(x)

        return x

def create_ResNet(num_classes=5, use_dropout=True, dropout_p=0.3):
    return ResNet(num_classes=num_classes, use_dropout=use_dropout, dropout_p=dropout_p)

def test_model():
    print("Creating ResNet model...")
    model = create_ResNet(num_classes=5)

    print("\n" + "="*50)
    print("ResNet MODEL ARCHITECTURE SUMMARY")
    print("="*50)

    try:
        summary(model, input_size=(1, 3, 224, 224), verbose=1)
    except Exception as e:
        print(f"Error in torchinfo summary: {e}")
        print("Trying manual forward pass...")

        model.eval()
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            output = model(test_input)
            print(f"Input shape: {test_input.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Expected output shape: (1, 5)")
            print(f"Model works correctly: {output.shape == (1, 5)}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model

print("✅ ResNet Model berhasil didefinisikan!")
print("📝 Model ini merupakan ResNet-34 dengan residual connections dan DROPOUT AKTIF")

print("🔧 Membuat model ResNet dengan dropout...")
model = create_ResNet(num_classes=5, use_dropout=True, dropout_p=0.3)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

print("=" * 60)
print("SUMMARY MODEL ResNet (DENGAN DROPOUT)")
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
    
    model.train()
    test_output_train = model(test_input)
    
    model.eval()
    test_output_eval = model(test_input)
    
    print(f"✓ Input shape: {test_input.shape}")
    print(f"✓ Output shape (train mode): {test_output_train.shape}")
    print(f"✓ Output shape (eval mode): {test_output_eval.shape}")
    print(f"✓ Output berbeda antara train/eval: {not torch.allclose(test_output_train, test_output_eval, atol=1e-5)}")
    print(f"✓ Forward pass berhasil!")

print("✅ Model ResNet dengan dropout aktif siap digunakan!")

BATCH_SIZE = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.9
EPOCHS = 10

print("🔄 Mempersiapkan data training...")
train_df = pd.read_csv('dataset/train.csv')
print(f"✓ Dataset dimuat: {len(train_df)} sampel")

train_data, val_data = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=42)
print(f"✓ Data split - Train: {len(train_data)}, Val: {len(val_data)}")

train_data.to_csv('train_split.csv', index=False)
val_data.to_csv('val_split.csv', index=False)
print("✓ Split data disimpan ke train_split.csv dan val_split.csv")

print("🔄 Membuat DataLoader...")
train_dataset = FoodDataset('train_split.csv', 'dataset/train', transform=train_transform)
val_dataset = FoodDataset('val_split.csv', 'dataset/train', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"✓ DataLoader siap - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

print("🔄 Menyiapkan loss function dan optimizer...")
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

"""
SGD Optimizer dengan Momentum dan Nesterov
- SGD (Stochastic Gradient Descent): optimizer dasar yang update weights berdasarkan gradient
- Momentum (0.9): menambahkan inersia dari gradient sebelumnya untuk mempercepat konvergensi
  dan mengurangi oscillation, seperti bola yang menggelinding menuruni bukit
- Nesterov=True: "look-ahead" momentum yang menghitung gradient di posisi yang akan datang
  memberikan update yang lebih akurat dan konvergensi lebih cepat
- Weight Decay (1e-4): L2 regularization untuk mencegah overfitting dengan menambah penalty
  pada magnitude weights yang besar
"""
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True, weight_decay=1e-4)

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

if EPOCHS > 5:
    base_cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS-5)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[LinearLR(optimizer, start_factor=0.1, total_iters=5), base_cosine],
        milestones=[5],
    )
else:
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

def train_model():
    print("🚀 Memulai training ResNet dengan DROPOUT AKTIF...")
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

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"{epoch+1:<6} {current_lr:<10.2e} {avg_train_loss:<12.4f} {train_acc:<12.4f} {avg_val_loss:<12.4f} {val_acc:<12.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_ckpt('resnet_best.pth', epoch+1, best_acc, train_losses, val_losses, train_accs, val_accs)
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

print("▶️  Memulai training ResNet dengan dropout...")
train_losses, val_losses, train_accs, val_accs = train_model()

print("\n✅ Training telah selesai! Metrics telah dicatat dan disimpan.")

print("📈 Membuat visualisasi metrik training...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

epochs_range = range(1, len(train_losses) + 1)

ax1.plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2)
ax1.plot(epochs_range, val_losses, 'r-', label='Validation Loss', linewidth=2)
ax1.set_title('Training dan Validation Loss (Dengan Dropout)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, train_accs, 'b-', label='Training Accuracy', linewidth=2)
ax2.plot(epochs_range, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
ax2.set_title('Training dan Validation Accuracy (Dengan Dropout)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

loss_diff = [val - train for val, train in zip(val_losses, train_losses)]
ax3.plot(epochs_range, loss_diff, 'g-', linewidth=2)
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax3.set_title('Loss Difference (Val - Train)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss Difference')
ax3.grid(True, alpha=0.3)
ax3.fill_between(epochs_range, loss_diff, alpha=0.3, color='green' if loss_diff[-1] < 0.5 else 'orange')

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

print("\n" + "="*60)
print("📊 RINGKASAN METRIK TRAINING ResNet (DENGAN DROPOUT)")
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

print(f"\n🔍 ANALYSIS (Dengan Dropout Regularization):")
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

print(f"\n🎯 DROPOUT EFFECTIVENESS:")
print(f"   Dropout membantu mengurangi overfitting dengan:")
print(f"   - Mengacak neuron selama training")
print(f"   - Mencegah co-adaptation antar neuron")
print(f"   - Meningkatkan robustness model")

print("="*60)

def test_single_batch():
    print("🔍 Testing DataLoader dengan single batch...")
    try:
        train_iter = iter(train_loader)
        images, labels = next(train_iter)
        print(f"✓ Train batch shape: {images.shape}, Labels shape: {labels.shape}")
        print(f"✓ Sample labels: {labels[:5].tolist()}")
        print(f"✓ Label classes: {[classes[i] for i in labels[:5]]}")

        val_iter = iter(val_loader)
        images, labels = next(val_iter)
        print(f"✓ Val batch shape: {images.shape}, Labels shape: {labels.shape}")

        images = images.to(device)
        
        model.train()
        with torch.no_grad():
            outputs_train = model(images)
            
        model.eval()
        with torch.no_grad():
            outputs_eval = model(images)
            
        print(f"✓ Model output shape: {outputs_train.shape}")
        print(f"✓ Sample predictions (train mode): {torch.argmax(outputs_train, dim=1)[:5].tolist()}")
        print(f"✓ Sample predictions (eval mode): {torch.argmax(outputs_eval, dim=1)[:5].tolist()}")
        print(f"✓ Output berbeda train vs eval: {not torch.allclose(outputs_train, outputs_eval, atol=1e-5)}")

        print("✅ DataLoader dan model test berhasil!")
        return True

    except Exception as e:
        print(f"❌ Error dalam test: {str(e)}")
        return False

test_result = test_single_batch()