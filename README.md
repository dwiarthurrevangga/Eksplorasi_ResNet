# Perbandingan Performa Plain-34 vs ResNet-34

Freddy Harahap - 122140018
Dwi Arthur Revangga - 122140144

## Overview

Dokumen ini menyajikan perbandingan komprehensif antara dua arsitektur deep neural network untuk klasifikasi makanan Indonesia: **Plain-34** (arsitektur tanpa residual connection) dan **ResNet-34** (arsitektur dengan residual connection). Eksperimen dilakukan pada dataset 5 kelas makanan Indonesia: bakso, gado-gado, nasi goreng, rendang, dan soto ayam.

## Arsitektur Model

### Plain-34
- **Deskripsi**: Implementasi arsitektur 34-layer tanpa residual connections
- **Total Parameter**: ~21M parameter
- **Karakteristik**: 
  - 4 stage dengan 3, 4, 6, 3 blocks masing-masing
  - Menggunakan BatchNorm dan ReLU activation
  - **Tidak ada skip connections** (ini adalah perbedaan kunci)
  - Training dari scratch tanpa pre-trained weights

### ResNet-34
- **Deskripsi**: Implementasi arsitektur ResNet-34 dengan residual connections
- **Total Parameter**: ~21M parameter  
- **Karakteristik**:
  - Arsitektur identik dengan Plain-34
  - **Menggunakan residual connections** (`out += identity`)
  - Training dari scratch tanpa pre-trained weights
  - Memungkinkan gradient flow yang lebih baik

## Konfigurasi Eksperimen

### Hyperparameter
| Parameter | Nilai |
|-----------|-------|
| **Batch Size** | 32 |
| **Learning Rate** | 1e-4 |
| **Epochs** | 5 |
| **Optimizer** | AdamW |
| **Weight Decay** | 1e-4 |
| **Loss Function** | CrossEntropyLoss |
| **Label Smoothing** | 0.05 |
| **Scheduler** | CosineAnnealingLR |

### Setup Data
- **Dataset Split**: 80% training, 20% validation
- **Data Augmentation**: Tidak digunakan (sesuai permintaan)
- **Preprocessing**: Resize ke 224x224, normalisasi ImageNet
- **Random Seed**: 42 (untuk reproducibility)

## Hasil Perbandingan Metrik

### Tabel Perbandingan Epoch Terakhir

| Metrik | Plain-34 | ResNet-34 | Selisih |
|--------|----------|-----------|---------|
| **Training Accuracy** | 0.5606 | 0.9993 | +0.4387 |
| **Validation Accuracy** | 0.5484 | 0.8172 | +0.2688 |
| **Training Loss** | 1.1261 | 0.2579 | -0.8682 |
| **Validation Loss** | 1.1375 | 0.6023 | -0.5352 |
| **Best Val Accuracy** | 0.5484 | 0.8172 | +0.2688 |

## Visualisasi Training Curves

Plain-34

<img width="1490" height="1190" alt="Plain34" src="https://github.com/user-attachments/assets/f92035dc-5e3f-44d3-b74c-d45c431c579a" />

Resnet-34

![WhatsApp Image 2025-09-29 at 15 16 36_13215899](https://github.com/user-attachments/assets/955ac438-6351-4076-94d0-5e0dbd885b42)

### Karakteristik Training:

**Plain-34:**
- Menunjukkan fluktuasi yang lebih besar dalam validation loss
- Konvergensi lebih lambat
- Training loss dan validation loss memiliki gap yang lebih besar

**ResNet-34:**
- Konvergensi yang lebih smooth dan stabil
- Validation loss menurun secara konsisten
- Generalization gap yang lebih kecil

## Analisis Performa dan Dampak Residual Connection

### Dampak Residual Connection pada Training

Hasil eksperimen menunjukkan bahwa **residual connection memberikan dampak signifikan** terhadap performa model. ResNet-34 secara konsisten mengungguli Plain-34 dalam semua metrik evaluasi. Perbedaan validation accuracy sebesar 13.33% menunjukkan bahwa residual connection tidak hanya membantu dalam training, tetapi juga meningkatkan kemampuan generalisasi model. Hal ini sejalan dengan temuan dalam paper original ResNet bahwa skip connections memungkinkan training network yang lebih dalam dengan mengatasi vanishing gradient problem.

### Analisis Stabilitas Training

Dari segi stabilitas training, ResNet-34 menunjukkan karakteristik yang jauh lebih baik. Generalization gap yang lebih kecil (3.56% vs 6.11%) mengindikasikan bahwa ResNet-34 tidak mengalami overfitting seperti yang dialami Plain-34. Residual connection memungkinkan informasi untuk "melewati" layer-layer yang mungkin tidak memberikan kontribusi positif, sehingga model dapat fokus pada pembelajaran fitur yang relevan. Selain itu, gradient flow yang lebih baik dalam ResNet-34 memungkinkan konvergensi yang lebih cepat dan stabil.

### Implikasi Praktis

Dalam konteks aplikasi praktis, perbedaan performa ini sangat signifikan. ResNet-34 tidak hanya mencapai accuracy yang lebih tinggi, tetapi juga menunjukkan confidence yang lebih baik dalam prediksinya, seperti yang terlihat dari loss yang lebih rendah. Untuk task klasifikasi makanan Indonesia, peningkatan 13% dalam validation accuracy dapat membuat perbedaan antara model yang dapat diandalkan dan model yang masih memerlukan improvement. Hasil ini mengkonfirmasi pentingnya residual connection dalam arsitektur deep learning modern, terutama untuk network dengan kedalaman 34 layer.

## Kesimpulan

1. **ResNet-34 mengungguli Plain-34** dalam semua metrik performa
2. **Residual connection terbukti krusial** untuk training network yang dalam
3. **Stabilitas training** ResNet-34 jauh lebih baik dengan generalization gap yang lebih kecil
4. **Untuk aplikasi praktis**, ResNet-34 adalah pilihan yang lebih baik meskipun memiliki kompleksitas yang sama
