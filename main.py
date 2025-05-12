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
    
    # Rastgele sayı üretecini ayarla
    set_random_seed(42)
    
    # Klasörleri oluştur
    setup_directories()
    
    # Orijinal veri setini kontrol et
    if not os.path.exists(IMAGES_DIR) or not os.path.exists(LABELS_DIR):
        print(f"HATA: Veri seti klasörleri bulunamadı: {IMAGES_DIR} veya {LABELS_DIR}")
        return
        
    # Orijinal görüntüleri kopyala
    copy_original_data()
    
    # Sınıf görüntülerini belirle - artık OUTPUT_IMAGES_DIR ve OUTPUT_LABELS_DIR kullanıyoruz
    class_images = get_class_images(OUTPUT_IMAGES_DIR, OUTPUT_LABELS_DIR)
    
    # Temel çoğaltmaları uygula
    apply_basic_augmentation(class_images)
    
    # Gelişmiş çoğaltmaları uygula
    apply_advanced_augmentation(class_images)
    
    # Çoğaltılmış veri setini analiz et
    analyze_augmented_dataset()
    
    # Veri setini eğitim ve doğrulama setlerine böl
    split_dataset()
    
    # YAML dosyası oluştur
    yaml_path = create_yaml_file()
    
    # Çoğaltma örneklerini görselleştir
    if VISUALIZE_EXAMPLES:
        visualize_augmentations()
    
    print("\n"+"="*80)
    print("Veri çoğaltma işlemi tamamlandı!")
    print("="*80)
    print("\nEğitim için aşağıdaki komutu kullanabilirsiniz:")
    print(f"python train.py --img 640 --batch 16 --epochs 300 --data {yaml_path} --weights yolo11l.pt")
    print("\nİki aşamalı eğitim için önerilen komutlar:")
    print("1. Aşama: Genel eğitim")
    print(f"python train.py --img 640 --batch 16 --epochs 200 --data {yaml_path} \\")
    print("    --weights yolo11l.pt --patience 50 --label-smoothing 0.1")
    print("\n2. Aşama: İnce ayar")
    print(f"python train.py --img 640 --batch 8 --epochs 100 --data {yaml_path} \\")
    print("    --weights runs/train/exp/weights/best.pt --patience 50 --freeze 10 --lr0 0.0001")

if __name__ == "__main__":
    main()