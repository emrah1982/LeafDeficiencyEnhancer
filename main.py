#!/usr/bin/env python
# main.py - Ana program

import os
import random
import numpy as np
from config import IMAGES_DIR, LABELS_DIR, OUTPUT_IMAGES_DIR, OUTPUT_LABELS_DIR, OUTPUT_DIR, VISUALIZE_EXAMPLES
from utils import get_class_images
from augmentation import copy_original_data, apply_basic_augmentation, apply_advanced_augmentation, analyze_augmented_dataset
from dataset import split_dataset, create_yaml_file
from visualization import visualize_augmentations

def setup_directories():
    """Gerekli dizinleri oluşturur"""
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)
    print(f"Klasörler oluşturuldu: {OUTPUT_DIR}")

def set_random_seed(seed=42):
    """Rastgele sayı üreticisini ayarlar"""
    random.seed(seed)
    np.random.seed(seed)
    print(f"Rastgele sayı üreteci ayarlandı: {seed}")

def main():
    """Ana program fonksiyonu"""
    print("="*80)
    print("Besin Eksikliği Veri Seti Çoğaltma Başlıyor...")
    print("="*80)
    
    # Rastgele sayı üreteci ayarla
    random.seed(42)
    print("Rastgele sayı üreteci ayarlandı: 42")
    
    # Klasörleri oluştur
    setup_directories()
    
    # Orijinal görüntüleri kopyala
    all_images = copy_original_data()
    
    # Her sınıf için görüntüleri belirle
    class_images = get_class_images(IMAGES_DIR, LABELS_DIR)
    
    # Temel çoğaltma uygula
    apply_basic_augmentation(class_images)
    
    # Mixup ve Mozaik çoğaltma uygula
    apply_advanced_augmentation(class_images)
    
    # Çoğaltılmış veri setini analiz et
    class_counts = analyze_augmented_dataset()
    
    # Veri setini train, val ve test olarak böl
    print("\nYOLO veri seti hazırlanıyor...")
    import split_dataset
    split_dataset.main()
    
    print("="*80)
    print("Veri çoğaltma işlemi tamamlandı!")
    print("="*80)
    
    # Eğitim seçeneklerini göster
    print("\nVeri çoğaltma işlemi tamamlandı!")
    print("Eğitim için train.py dosyasını çalıştırın.")

if __name__ == "__main__":
    main()