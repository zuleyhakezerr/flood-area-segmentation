# 🌊 Sel Alanı Segmentasyonu - Ödev 5

Bu proje, uydu/hava fotoğrafları üzerinden sel basmış alanları piksel düzeyinde tespit etmeyi (semantic segmentation) amaçlamaktadır.

## 📋 İçerik

- **Veri Seti**: 290 adet havadan çekilmiş sel görüntüsü ve bunlara ait binary maskeler
- **Problem Tipi**: Semantik Segmentasyon (Piksel düzeyinde sınıflandırma: Su mu, Kara mı?)
- **Kullanılan Framework**: TensorFlow/Keras

## 🏗️ Model Mimarileri

| Model | Açıklama |
|-------|----------|
| **U-Net** | Encoder-Decoder yapısı ile skip connections. Biyomedikal segmentasyon için geliştirilmiş klasik mimari. |
| **SegNet** | Encoder-Decoder with pooling indices. Sürücüsüz araç sistemleri için optimize edilmiş. |
| **FPN** | Feature Pyramid Network. Çoklu ölçekli özellik haritaları kullanır. |
| **DeepLabV3+** | Atrous Spatial Pyramid Pooling (ASPP) + Decoder. Google'ın state-of-the-art mimarisi. |
| **EfficientNet-UNet** | Transfer learning ile güçlendirilmiş U-Net. |

## 📊 Değerlendirme Metrikleri

- **Dice Coefficient**: F1 Score benzeri, overlap ölçümü
- **IoU (Jaccard Index)**: Intersection over Union
- **Binary Crossentropy**: Piksel bazlı kayıp
- **Combined Loss**: BCE + Dice Loss kombinasyonu

## 🚀 Kurulum

```bash
# Gerekli kütüphaneleri yükle
pip install -r requirements.txt
```

## 💻 Kullanım

### Tüm Modelleri Eğitmek İçin:
```python
python flood_segmentation.py
```

### Tek Model Eğitimi (Hızlı Test):
```python
from flood_segmentation import train_single_model

# U-Net modelini 20 epoch eğit
model, results, history = train_single_model('U-Net', epochs=20)
```

### Sadece Belirli Modeli Eğitmek:
```python
from flood_segmentation import *

# Veri yükle
image_files, mask_files = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(image_files, mask_files)

# Dataset oluştur
train_dataset = create_dataset(X_train, y_train, augment_data=True)
val_dataset = create_dataset(X_val, y_val, augment_data=False)
test_dataset = create_dataset(X_test, y_test, augment_data=False)

# Model seç ve eğit
model = build_deeplabv3plus()  # veya build_unet(), build_segnet(), build_fpn()
model = compile_model(model)
history = train_model(model, train_dataset, val_dataset, 'DeepLabV3+', epochs=30)

# Değerlendir ve görselleştir
evaluate_model(model, test_dataset, 'DeepLabV3+')
visualize_predictions(model, test_dataset, 'DeepLabV3+')
```

## 📁 Dosya Yapısı

```
sel ödev/
├── archive/
│   ├── Image/          # Sel görüntüleri (*.jpg)
│   ├── Mask/           # Segmentasyon maskeleri (*.png)
│   └── metadata.csv    # Görüntü-mask eşleştirmesi
├── flood_segmentation.py   # Ana kod
├── requirements.txt        # Gerekli kütüphaneler
└── README.md              # Bu dosya
```

## 📈 Çıktılar

Eğitim sonrası aşağıdaki dosyalar oluşturulur:

- `best_{model_name}.keras` - En iyi model ağırlıkları
- `{model_name}_training_history.png` - Eğitim grafikleri
- `{model_name}_predictions.png` - Tahmin karşılaştırmaları
- `{model_name}_overlay.png` - Overlay görselleştirmeler
- `model_comparison.png` - Tüm modellerin karşılaştırması

## ⚙️ Konfigürasyon

`Config` sınıfından parametreleri değiştirebilirsiniz:

```python
class Config:
    IMG_HEIGHT = 256      # Görüntü yüksekliği
    IMG_WIDTH = 256       # Görüntü genişliği
    BATCH_SIZE = 8        # Batch boyutu
    EPOCHS = 50           # Eğitim epoch sayısı
    LEARNING_RATE = 1e-4  # Öğrenme oranı
    VAL_SPLIT = 0.15      # Validation oranı
    TEST_SPLIT = 0.15     # Test oranı
```

## 📝 Notlar

- GPU kullanımı önerilir (eğitim CPU'da çok yavaş olabilir)
- Bellek yetersizliği durumunda `BATCH_SIZE` değerini düşürün
- Veri artırma (augmentation) eğitim setine otomatik uygulanır
- Early stopping ile overfitting önlenir

## 👤 Geliştirici

Züleyha - Görüntü İşleme Ödevi 5


