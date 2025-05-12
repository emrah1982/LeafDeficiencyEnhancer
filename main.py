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
    # YOLO modellerini indirme seçenekleri
    print("\nYOLO modellerini indirmek ister misiniz?")
    print("1. Tüm modelleri indir (YOLOv8n, YOLOv8s, YOLOv8m)")
    print("2. Sadece seçilen modeli indir")
    print("3. Model indirme (zaten indirilmiş)")
    
    while True:
        download_choice = input("\nSeçiminiz (1/2/3): ")
        if download_choice in ['1', '2', '3']:
            break
        print("Lütfen 1, 2 veya 3 girin.")
    
    if download_choice == '1':
        print("\nTüm modeller indiriliyor...")
        os.system("yolo download model yolov8n.pt")
        os.system("yolo download model yolov8s.pt")
        os.system("yolo download model yolov8m.pt")
        print("Tüm modeller indirildi!")
    elif download_choice == '2':
        print("\nHangi modeli indirmek istersiniz?")
        print("1. YOLOv8n (küçük model)")
        print("2. YOLOv8s (orta model)")
        print("3. YOLOv8m (büyük model)")
        
        while True:
            model_choice = input("\nModel seçiminiz (1/2/3): ")
            if model_choice in ['1', '2', '3']:
                break
            print("Lütfen 1, 2 veya 3 girin.")
        
        models = {
            '1': 'yolov8n.pt',
            '2': 'yolov8s.pt',
            '3': 'yolov8m.pt'
        }
        
        print(f"\n{models[model_choice]} indiriliyor...")
        os.system(f"yolo download model {models[model_choice]}")
        print("Model indirildi!")
    
    # Eğitim seçenekleri
    print("\nEğitim için seçenekler:")
    print("1. Hızlı Eğitim (YOLOv8n - 100 epoch)")
    print("2. Standart Eğitim (YOLOv8s - 200 epoch)")
    print("3. Detaylı Eğitim (YOLOv8m - 300 epoch)")
    print("4. İki Aşamalı Eğitim (YOLOv8s - Ön eğitim + İnce ayar)")
    
    while True:
        choice = input("\nHangi eğitim modelini kullanmak istersiniz? (1/2/3/4): ")
        if choice in ['1', '2', '3', '4']:
            break
        print("Lütfen 1, 2, 3 veya 4 girin.")
    
    if choice == '1':
        print("\nHızlı Eğitim komutu:")
        print("yolo task=detect mode=train model=yolov8n.pt data=dataset/yolo/dataset.yaml epochs=100 imgsz=640 batch=16")
    
    elif choice == '2':
        print("\nStandart Eğitim komutu:")
        print("yolo task=detect mode=train model=yolov8s.pt data=dataset/yolo/dataset.yaml epochs=200 imgsz=640 batch=16 patience=50")
    
    elif choice == '3':
        print("\nDetaylı Eğitim komutu:")
        print("yolo task=detect mode=train model=yolov8m.pt data=dataset/yolo/dataset.yaml epochs=300 imgsz=640 batch=16 patience=50")
    
    else:
        print("\nİki Aşamalı Eğitim komutları:")
        print("1. Aşama - Ön eğitim:")
        print("yolo task=detect mode=train model=yolov8s.pt data=dataset/yolo/dataset.yaml epochs=200 imgsz=640 batch=16 patience=50 label-smoothing=0.1")
        print("\n2. Aşama - İnce ayar (ilk aşama tamamlandıktan sonra):")
        print("yolo task=detect mode=train model=runs/train/exp/weights/best.pt data=dataset/yolo/dataset.yaml epochs=100 imgsz=640 batch=8 patience=50 freeze=10")
    
    # İleri seviye eğitim önerileri
    print("\nÖneriler:")
    print("1. İlk önce küçük bir model (yolov8n.pt) ile kısa bir eğitim (10-20 epoch) yaparak")
    print("   her şeyin doğru çalıştığından emin olun.")
    print("2. Daha sonra daha büyük bir model (yolov8s.pt veya yolov8m.pt) ve daha fazla")
    print("   epoch (100-300) ile tam eğitimi gerçekleştirin.")
    print("3. Eğitim sırasında runs/train/besin_eksikligi klasöründe sonuçları ve")
    print("   grafikleri görebilirsiniz.")
    print("4. En iyi model runs/train/besin_eksikligi/weights/best.pt olarak kaydedilecektir.")

if __name__ == "__main__":
    main()