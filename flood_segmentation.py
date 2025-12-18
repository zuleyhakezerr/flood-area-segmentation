# =============================================================================
# SEL ALANI SEGMENTASYONU - Ödev 5
# Semantik Segmentasyon (U-Net, SegNet, FPN, DeepLabV3+)
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0, ResNet50

# GPU bellek ayarı (varsa)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✓ {len(gpus)} GPU bulundu!")
else:
    print("⚠ GPU bulunamadı, CPU kullanılacak.")

# =============================================================================
# 1. YAPILANDIRMA (CONFIGURATION)
# =============================================================================

class Config:
    """Model ve eğitim parametreleri"""
    # Veri yolları
    BASE_PATH = "/Users/zuleyha/sel ödev/archive"
    IMAGE_PATH = os.path.join(BASE_PATH, "Image")
    MASK_PATH = os.path.join(BASE_PATH, "Mask")
    METADATA_PATH = os.path.join(BASE_PATH, "metadata.csv")
    
    # Model parametreleri - Küçük boyut = Hızlı eğitim
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    IMG_CHANNELS = 3
    
    # Eğitim parametreleri - Hızlı eğitim için optimize edildi
    BATCH_SIZE = 16
    EPOCHS = 15
    LEARNING_RATE = 1e-3  # Daha yüksek LR = daha hızlı öğrenme
    
    # Veri bölme oranları
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15

config = Config()

# =============================================================================
# 2. VERİ YÜKLEME VE HAZIRLAMA
# =============================================================================

def load_data():
    """Metadata'dan görüntü ve mask yollarını yükle"""
    metadata = pd.read_csv(config.METADATA_PATH)
    print(f"✓ Toplam veri sayısı: {len(metadata)}")
    
    image_files = [os.path.join(config.IMAGE_PATH, f) for f in metadata['Image'].values]
    mask_files = [os.path.join(config.MASK_PATH, f) for f in metadata['Mask'].values]
    
    return image_files, mask_files

def split_data(image_files, mask_files):
    """Veriyi train/validation/test olarak böl"""
    # İlk önce test setini ayır
    X_train, X_test, y_train, y_test = train_test_split(
        image_files, mask_files, 
        test_size=config.TEST_SPLIT, 
        random_state=42
    )
    
    # Kalan veriden validation setini ayır
    val_ratio = config.VAL_SPLIT / (1 - config.TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=val_ratio, 
        random_state=42
    )
    
    print(f"✓ Eğitim seti: {len(X_train)} görüntü")
    print(f"✓ Doğrulama seti: {len(X_val)} görüntü")
    print(f"✓ Test seti: {len(X_test)} görüntü")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# =============================================================================
# 3. VERİ ARTIRMA (DATA AUGMENTATION)
# =============================================================================

def load_image(image_path, mask_path):
    """Görüntü ve maskeyi yükle, normalize et"""
    # Görüntü yükleme
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.ensure_shape(image, [None, None, 3])
    image = tf.image.resize(image, [config.IMG_HEIGHT, config.IMG_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0
    
    # Mask yükleme
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
    mask = tf.ensure_shape(mask, [None, None, 1])
    mask = tf.image.resize(mask, [config.IMG_HEIGHT, config.IMG_WIDTH], method='nearest')
    mask = tf.cast(mask, tf.float32) / 255.0
    
    return image, mask

def augment(image, mask):
    """Veri artırma fonksiyonu"""
    # Rastgele yatay çevirme
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    
    # Rastgele dikey çevirme
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    
    # Rastgele 90 derece döndürme
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    mask = tf.image.rot90(mask, k)
    
    # Rastgele parlaklık (sadece görüntüye)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, mask

def create_dataset(image_paths, mask_paths, augment_data=False, batch_size=None):
    """TensorFlow Dataset oluştur"""
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment_data:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# =============================================================================
# 4. KAYIP FONKSİYONLARI VE METRİKLER
# =============================================================================

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice katsayısı (F1 Score benzeri)"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice Loss = 1 - Dice Coefficient"""
    return 1 - dice_coefficient(y_true, y_pred)

def iou_score(y_true, y_pred, smooth=1e-6):
    """IoU (Intersection over Union / Jaccard Index)"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def combined_loss(y_true, y_pred):
    """Binary Crossentropy + Dice Loss kombinasyonu"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

# =============================================================================
# 5. MODEL MİMARİLERİ
# =============================================================================

# ----- 5.1 U-NET -----
def build_unet(input_shape=(128, 128, 3)):
    """
    U-Net: Encoder-Decoder yapısı ile skip connections
    Biyomedikal görüntü segmentasyonu için geliştirilmiş klasik mimari
    """
    inputs = layers.Input(shape=input_shape)
    
    # ========== ENCODER (Sıkıştırma Yolu) ==========
    # Block 1
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D(2)(c1)
    
    # Block 2
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D(2)(c2)
    
    # Block 3
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D(2)(c3)
    
    # Block 4
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p3)
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Dropout(0.3)(c4)
    p4 = layers.MaxPooling2D(2)(c4)
    
    # ========== BOTTLENECK (En derin nokta) ==========
    c5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p4)
    c5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Dropout(0.3)(c5)
    
    # ========== DECODER (Genişleme Yolu + Skip Connections) ==========
    # Block 6
    u6 = layers.Conv2DTranspose(512, 2, strides=2, padding='same')(c5)
    u6 = layers.concatenate([u6, c4])  # Skip connection
    c6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u6)
    c6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c6)
    c6 = layers.BatchNormalization()(c6)
    
    # Block 7
    u7 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u7)
    c7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c7)
    c7 = layers.BatchNormalization()(c7)
    
    # Block 8
    u8 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u8)
    c8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c8)
    c8 = layers.BatchNormalization()(c8)
    
    # Block 9
    u9 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u9)
    c9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c9)
    c9 = layers.BatchNormalization()(c9)
    
    # ========== ÇIKIŞ KATMANI ==========
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c9)
    
    model = Model(inputs, outputs, name='U-Net')
    return model


# ----- 5.2 SEGNET -----
def build_segnet(input_shape=(128, 128, 3)):
    """
    SegNet: Encoder-Decoder with pooling indices
    Sürücüsüz araç sistemleri için geliştirilmiş mimari
    """
    inputs = layers.Input(shape=input_shape)
    
    # ========== ENCODER ==========
    # Block 1
    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2, strides=2)(x)
    
    # Block 2
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2, strides=2)(x)
    
    # Block 3
    x = layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2, strides=2)(x)
    
    # Block 4
    x = layers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2, strides=2)(x)
    
    # ========== DECODER ==========
    # Block 5
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Block 6
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Block 7
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Block 8
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # ========== ÇIKIŞ KATMANI ==========
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs, name='SegNet')
    return model


# ----- 5.3 FPN (Feature Pyramid Network) -----
def build_fpn(input_shape=(128, 128, 3)):
    """
    FPN: Feature Pyramid Network
    Çoklu ölçekli özellik haritaları kullanarak segmentasyon
    """
    inputs = layers.Input(shape=input_shape)
    
    # ========== BACKBONE (ResNet benzeri) ==========
    # Stage 1
    x = layers.Conv2D(64, 7, strides=2, padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    c1 = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Stage 2
    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(c1)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    c2 = layers.Activation('relu')(x)
    
    # Stage 3
    x = layers.Conv2D(128, 3, strides=2, padding='same', kernel_initializer='he_normal')(c2)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    c3 = layers.Activation('relu')(x)
    
    # Stage 4
    x = layers.Conv2D(256, 3, strides=2, padding='same', kernel_initializer='he_normal')(c3)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    c4 = layers.Activation('relu')(x)
    
    # Stage 5
    x = layers.Conv2D(512, 3, strides=2, padding='same', kernel_initializer='he_normal')(c4)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    c5 = layers.Activation('relu')(x)
    
    # ========== FPN LATERAL CONNECTIONS ==========
    # Tüm feature haritalarını 256 kanala dönüştür
    p5 = layers.Conv2D(256, 1, padding='same', kernel_initializer='he_normal')(c5)
    
    p4 = layers.Conv2D(256, 1, padding='same', kernel_initializer='he_normal')(c4)
    p4 = layers.Add()([layers.UpSampling2D(2)(p5), p4])
    
    p3 = layers.Conv2D(256, 1, padding='same', kernel_initializer='he_normal')(c3)
    p3 = layers.Add()([layers.UpSampling2D(2)(p4), p3])
    
    p2 = layers.Conv2D(256, 1, padding='same', kernel_initializer='he_normal')(c2)
    p2 = layers.Add()([layers.UpSampling2D(2)(p3), p2])
    
    # ========== SEGMENTASYON HEAD ==========
    # Tüm piramit seviyelerini birleştir
    p5_up = layers.UpSampling2D(8)(p5)
    p4_up = layers.UpSampling2D(4)(p4)
    p3_up = layers.UpSampling2D(2)(p3)
    p2_up = p2
    
    merged = layers.Concatenate()([p2_up, p3_up, p4_up, p5_up])
    
    x = layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Orijinal boyuta upsample
    x = layers.UpSampling2D(4)(x)
    
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs, name='FPN')
    return model


# ----- 5.4 DeepLabV3+ -----
def build_deeplabv3plus(input_shape=(128, 128, 3)):
    """
    DeepLabV3+: Atrous Spatial Pyramid Pooling (ASPP) + Decoder
    Google tarafından geliştirilen state-of-the-art segmentasyon mimarisi
    """
    inputs = layers.Input(shape=input_shape)
    
    # ========== ENCODER (Basitleştirilmiş) ==========
    # Initial Conv
    x = layers.Conv2D(64, 3, strides=2, padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Block 1
    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    low_level_features = x  # Düşük seviye özellikler (decoder için)
    
    # Block 2
    x = layers.Conv2D(128, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Block 3
    x = layers.Conv2D(256, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # ========== ASPP (Atrous Spatial Pyramid Pooling) ==========
    # 1x1 Conv
    aspp1 = layers.Conv2D(256, 1, padding='same', kernel_initializer='he_normal')(x)
    aspp1 = layers.BatchNormalization()(aspp1)
    aspp1 = layers.Activation('relu')(aspp1)
    
    # 3x3 Conv, rate=6
    aspp2 = layers.Conv2D(256, 3, padding='same', dilation_rate=6, kernel_initializer='he_normal')(x)
    aspp2 = layers.BatchNormalization()(aspp2)
    aspp2 = layers.Activation('relu')(aspp2)
    
    # 3x3 Conv, rate=12
    aspp3 = layers.Conv2D(256, 3, padding='same', dilation_rate=12, kernel_initializer='he_normal')(x)
    aspp3 = layers.BatchNormalization()(aspp3)
    aspp3 = layers.Activation('relu')(aspp3)
    
    # 3x3 Conv, rate=18
    aspp4 = layers.Conv2D(256, 3, padding='same', dilation_rate=18, kernel_initializer='he_normal')(x)
    aspp4 = layers.BatchNormalization()(aspp4)
    aspp4 = layers.Activation('relu')(aspp4)
    
    # Image Pooling
    aspp5 = layers.GlobalAveragePooling2D()(x)
    aspp5 = layers.Reshape((1, 1, 256))(aspp5)
    aspp5 = layers.Conv2D(256, 1, padding='same', kernel_initializer='he_normal')(aspp5)
    aspp5 = layers.BatchNormalization()(aspp5)
    aspp5 = layers.Activation('relu')(aspp5)
    aspp5 = layers.UpSampling2D(size=(x.shape[1], x.shape[2]), interpolation='bilinear')(aspp5)
    
    # ASPP çıktılarını birleştir
    aspp_out = layers.Concatenate()([aspp1, aspp2, aspp3, aspp4, aspp5])
    aspp_out = layers.Conv2D(256, 1, padding='same', kernel_initializer='he_normal')(aspp_out)
    aspp_out = layers.BatchNormalization()(aspp_out)
    aspp_out = layers.Activation('relu')(aspp_out)
    aspp_out = layers.Dropout(0.5)(aspp_out)
    
    # ========== DECODER ==========
    # ASPP çıktısını upsample
    x = layers.UpSampling2D(4, interpolation='bilinear')(aspp_out)
    
    # Low-level features işleme
    low_level = layers.Conv2D(48, 1, padding='same', kernel_initializer='he_normal')(low_level_features)
    low_level = layers.BatchNormalization()(low_level)
    low_level = layers.Activation('relu')(low_level)
    
    # Birleştir
    x = layers.Concatenate()([x, low_level])
    
    # Final Conv
    x = layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Orijinal boyuta upsample
    x = layers.UpSampling2D(2, interpolation='bilinear')(x)
    
    # ========== ÇIKIŞ KATMANI ==========
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs, name='DeepLabV3Plus')
    return model


# ----- 5.5 EfficientNet U-Net -----
def build_efficientnet_unet(input_shape=(128, 128, 3)):
    """
    EfficientNet tabanlı U-Net
    Transfer learning ile güçlendirilmiş mimari
    """
    # EfficientNetB0'ı backbone olarak kullan
    backbone = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Encoder çıktıları (skip connections için)
    skip_names = [
        'block2a_expand_activation',  # 128x128
        'block3a_expand_activation',  # 64x64
        'block4a_expand_activation',  # 32x32
        'block6a_expand_activation',  # 16x16
    ]
    
    skip_outputs = [backbone.get_layer(name).output for name in skip_names]
    
    # Decoder
    x = backbone.output
    
    # Decoder Block 1
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, skip_outputs[3]])
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Decoder Block 2
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, skip_outputs[2]])
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Decoder Block 3
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, skip_outputs[1]])
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Decoder Block 4
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, skip_outputs[0]])
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Final upsample
    x = layers.UpSampling2D(2)(x)
    
    # Çıkış
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)
    
    model = Model(backbone.input, outputs, name='EfficientNet-UNet')
    return model

# =============================================================================
# 6. MODEL DERLEME VE EĞİTİM
# =============================================================================

def compile_model(model, learning_rate=None):
    """Modeli derle"""
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=combined_loss,
        metrics=[dice_coefficient, iou_score, 'accuracy']
    )
    return model

def get_callbacks(model_name):
    """Eğitim callback'lerini oluştur - Hızlı eğitim için optimize edildi"""
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,  # 3 epoch iyileşme yoksa dur
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,  # 2 epoch bekle
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            f'best_{model_name}.keras',
            monitor='val_iou_score',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks

def train_model(model, train_dataset, val_dataset, model_name, epochs=None):
    """Modeli eğit"""
    if epochs is None:
        epochs = config.EPOCHS
    
    print(f"\n{'='*60}")
    print(f"🚀 {model_name} Eğitimi Başlıyor...")
    print(f"{'='*60}\n")
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=get_callbacks(model_name),
        verbose=1
    )
    
    return history

# =============================================================================
# 7. DEĞERLENDİRME VE GÖRSELLEŞTİRME
# =============================================================================

def evaluate_model(model, test_dataset, model_name):
    """Model performansını değerlendir"""
    print(f"\n📊 {model_name} Test Değerlendirmesi:")
    results = model.evaluate(test_dataset, verbose=0)
    
    metrics = ['Loss', 'Dice Coefficient', 'IoU Score', 'Accuracy']
    for metric, value in zip(metrics, results):
        print(f"   {metric}: {value:.4f}")
    
    return dict(zip(metrics, results))

def plot_training_history(history, model_name):
    """Eğitim grafiklerini çiz"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Eğitim', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Doğrulama', linewidth=2)
    axes[0, 0].set_title(f'{model_name} - Kayıp (Loss)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Dice Coefficient
    axes[0, 1].plot(history.history['dice_coefficient'], label='Eğitim', linewidth=2)
    axes[0, 1].plot(history.history['val_dice_coefficient'], label='Doğrulama', linewidth=2)
    axes[0, 1].set_title(f'{model_name} - Dice Coefficient', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # IoU Score
    axes[1, 0].plot(history.history['iou_score'], label='Eğitim', linewidth=2)
    axes[1, 0].plot(history.history['val_iou_score'], label='Doğrulama', linewidth=2)
    axes[1, 0].set_title(f'{model_name} - IoU Score (Jaccard)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 1].plot(history.history['accuracy'], label='Eğitim', linewidth=2)
    axes[1, 1].plot(history.history['val_accuracy'], label='Doğrulama', linewidth=2)
    axes[1, 1].set_title(f'{model_name} - Doğruluk (Accuracy)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Grafik kaydedildi: {model_name}_training_history.png")

def visualize_predictions(model, test_dataset, model_name, num_samples=5):
    """Model tahminlerini gerçek maskelerle karşılaştır"""
    # Test verisinden birkaç örnek al
    for images, masks in test_dataset.take(1):
        predictions = model.predict(images, verbose=0)
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        
        for i in range(min(num_samples, len(images))):
            # Orijinal görüntü
            axes[i, 0].imshow(images[i].numpy())
            axes[i, 0].set_title('Orijinal Görüntü', fontsize=10)
            axes[i, 0].axis('off')
            
            # Gerçek mask
            axes[i, 1].imshow(masks[i].numpy().squeeze(), cmap='Blues')
            axes[i, 1].set_title('Gerçek Mask (Ground Truth)', fontsize=10)
            axes[i, 1].axis('off')
            
            # Tahmin edilen mask (ham)
            axes[i, 2].imshow(predictions[i].squeeze(), cmap='Blues')
            axes[i, 2].set_title('Tahmin (Ham)', fontsize=10)
            axes[i, 2].axis('off')
            
            # Tahmin edilen mask (eşiklenmiş)
            pred_binary = (predictions[i].squeeze() > 0.5).astype(np.float32)
            axes[i, 3].imshow(pred_binary, cmap='Blues')
            axes[i, 3].set_title('Tahmin (Eşiklenmiş > 0.5)', fontsize=10)
            axes[i, 3].axis('off')
        
        plt.suptitle(f'{model_name} - Tahmin Sonuçları', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{model_name}_predictions.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✓ Tahmin görseli kaydedildi: {model_name}_predictions.png")

def visualize_overlay(model, test_dataset, model_name, num_samples=3):
    """Tahminleri orijinal görüntü üzerine bindirerek göster"""
    for images, masks in test_dataset.take(1):
        predictions = model.predict(images, verbose=0)
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
        
        for i in range(min(num_samples, len(images))):
            img = images[i].numpy()
            true_mask = masks[i].numpy().squeeze()
            pred_mask = (predictions[i].squeeze() > 0.5).astype(np.float32)
            
            # Orijinal görüntü
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Orijinal Görüntü', fontsize=11)
            axes[i, 0].axis('off')
            
            # Gerçek mask overlay
            axes[i, 1].imshow(img)
            axes[i, 1].imshow(true_mask, alpha=0.5, cmap='Reds')
            axes[i, 1].set_title('Gerçek Sel Alanı (Kırmızı)', fontsize=11)
            axes[i, 1].axis('off')
            
            # Tahmin overlay
            axes[i, 2].imshow(img)
            axes[i, 2].imshow(pred_mask, alpha=0.5, cmap='Blues')
            axes[i, 2].set_title('Tahmin Edilen Sel Alanı (Mavi)', fontsize=11)
            axes[i, 2].axis('off')
        
        plt.suptitle(f'{model_name} - Overlay Görselleştirme', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{model_name}_overlay.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✓ Overlay görseli kaydedildi: {model_name}_overlay.png")

def compare_models(results_dict):
    """Tüm modellerin performansını karşılaştır"""
    models = list(results_dict.keys())
    metrics = ['Dice Coefficient', 'IoU Score', 'Accuracy']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    
    for idx, metric in enumerate(metrics):
        values = [results_dict[model][metric] for model in models]
        bars = axes[idx].bar(models, values, color=colors[:len(models)], edgecolor='black', linewidth=1.2)
        axes[idx].set_title(metric, fontsize=12, fontweight='bold')
        axes[idx].set_ylim(0, 1)
        axes[idx].set_ylabel('Skor')
        
        # Değerleri barların üstüne yaz
        for bar, val in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                          f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Model Karşılaştırması', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Karşılaştırma grafiği kaydedildi: model_comparison.png")

# =============================================================================
# 8. ANA PROGRAM
# =============================================================================

def main():
    """Ana çalıştırma fonksiyonu"""
    print("\n" + "="*70)
    print("🌊 SEL ALANI SEGMENTASYONU - Ödev 5")
    print("="*70 + "\n")
    
    # 1. Veri Yükleme
    print("📁 Veri yükleniyor...")
    image_files, mask_files = load_data()
    
    # 2. Veri Bölme
    print("\n📊 Veri bölünüyor...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(image_files, mask_files)
    
    # 3. Dataset Oluşturma
    print("\n🔄 Dataset'ler oluşturuluyor...")
    train_dataset = create_dataset(X_train, y_train, augment_data=True)
    val_dataset = create_dataset(X_val, y_val, augment_data=False)
    test_dataset = create_dataset(X_test, y_test, augment_data=False)
    print("✓ Dataset'ler hazır!")
    
    # 4. Modelleri Tanımla (Hızlı eğitim için 2 model)
    models = {
        'U-Net': build_unet,
        'DeepLabV3+': build_deeplabv3plus,
    }
    
    # Sonuçları sakla
    all_results = {}
    all_histories = {}
    trained_models = {}
    
    # 5. Her modeli eğit ve değerlendir
    for model_name, model_builder in models.items():
        print(f"\n{'='*70}")
        print(f"🏗️ {model_name} modeli oluşturuluyor...")
        
        # Model oluştur
        model = model_builder()
        model = compile_model(model)
        
        # Model özeti
        print(f"\n📋 {model_name} Özeti:")
        print(f"   Toplam parametre: {model.count_params():,}")
        
        # Eğitim
        history = train_model(model, train_dataset, val_dataset, model_name)
        all_histories[model_name] = history
        trained_models[model_name] = model
        
        # Değerlendirme
        results = evaluate_model(model, test_dataset, model_name)
        all_results[model_name] = results
        
        # Görselleştirme
        plot_training_history(history, model_name)
        visualize_predictions(model, test_dataset, model_name)
        visualize_overlay(model, test_dataset, model_name)
    
    # 6. Model Karşılaştırması
    print("\n" + "="*70)
    print("📈 Model Karşılaştırması")
    print("="*70)
    compare_models(all_results)
    
    # En iyi modeli göster
    best_model = max(all_results, key=lambda x: all_results[x]['IoU Score'])
    print(f"\n🏆 En İyi Model: {best_model}")
    print(f"   IoU Score: {all_results[best_model]['IoU Score']:.4f}")
    print(f"   Dice Coefficient: {all_results[best_model]['Dice Coefficient']:.4f}")
    
    print("\n✅ Tüm işlemler tamamlandı!")
    
    return trained_models, all_results, all_histories

# =============================================================================
# 9. TEK MODEL EĞİTİMİ (Hızlı Test İçin)
# =============================================================================

def train_single_model(model_name='U-Net', epochs=20):
    """Sadece tek bir modeli eğit (hızlı test için)"""
    print("\n" + "="*70)
    print(f"🌊 SEL ALANI SEGMENTASYONU - {model_name}")
    print("="*70 + "\n")
    
    # Veri Yükleme
    print("📁 Veri yükleniyor...")
    image_files, mask_files = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(image_files, mask_files)
    
    # Dataset Oluşturma
    train_dataset = create_dataset(X_train, y_train, augment_data=True)
    val_dataset = create_dataset(X_val, y_val, augment_data=False)
    test_dataset = create_dataset(X_test, y_test, augment_data=False)
    
    # Model Oluşturma
    model_builders = {
        'U-Net': build_unet,
        'SegNet': build_segnet,
        'FPN': build_fpn,
        'DeepLabV3+': build_deeplabv3plus,
        'EfficientNet-UNet': build_efficientnet_unet,
    }
    
    model = model_builders[model_name]()
    model = compile_model(model)
    print(f"\n📋 {model_name} - Toplam parametre: {model.count_params():,}")
    
    # Eğitim
    history = train_model(model, train_dataset, val_dataset, model_name, epochs=epochs)
    
    # Değerlendirme
    results = evaluate_model(model, test_dataset, model_name)
    
    # Görselleştirme
    plot_training_history(history, model_name)
    visualize_predictions(model, test_dataset, model_name)
    visualize_overlay(model, test_dataset, model_name)
    
    return model, results, history

# =============================================================================
# 10. ÇALIŞTIR
# =============================================================================

if __name__ == "__main__":
    # Tek model eğitimi (hızlı test için)
    # model, results, history = train_single_model('U-Net', epochs=20)
    
    # Tüm modelleri eğit ve karşılaştır
    trained_models, all_results, all_histories = main()

