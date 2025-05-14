# augmentation.py - Veri çoğaltma işlemleri

import os
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import CLASS_INFO, IMAGES_DIR, LABELS_DIR, OUTPUT_IMAGES_DIR, OUTPUT_LABELS_DIR, OUTPUT_DIR
from utils import yolo_to_bbox, bbox_to_yolo, create_mixup, create_mosaic
from transforms import get_transform_by_class

def copy_original_data():
    """Orijinal görüntüleri çıktı klasörüne kopyalar"""
    print("Orijinal görüntüler kopyalanıyor...")
    all_images = []
    
    # Görüntü klasörünü kontrol et
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'))]
    
    for img_file in image_files:
        img_path = os.path.join(IMAGES_DIR, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(LABELS_DIR, label_file)
        
        if os.path.exists(label_path):
            # Orijinal dosyaları kopyala
            shutil.copy(img_path, os.path.join(OUTPUT_IMAGES_DIR, img_file))
            shutil.copy(label_path, os.path.join(OUTPUT_LABELS_DIR, label_file))
            all_images.append(img_file)
    
    print(f"{len(all_images)} adet orijinal görüntü kopyalandı.")
    return all_images

def apply_basic_augmentation(class_images):
    """Temel veri çoğaltma işlemlerini uygular"""
    print("Temel çoğaltma uygulanıyor...")
    
    # Sadece görüntü sayısı sıfırdan büyük olan sınıflar için işlem yap
    active_classes = {class_id: info for class_id, info in CLASS_INFO.items() if info['count'] > 0}
    
    for class_id, images in class_images.items():
        augmentation_factor = CLASS_INFO[class_id]["target"] // CLASS_INFO[class_id]["count"] + 1
        
        print(f"Sınıf {class_id} ({CLASS_INFO[class_id]['name']}) için çoğaltma faktörü: {augmentation_factor}x")
        
        for img_file in tqdm(images, desc=f"Sınıf {class_id} çoğaltılıyor"):
            img_path = os.path.join(IMAGES_DIR, img_file)
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(LABELS_DIR, label_file)
            
            # Görüntüyü oku
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_height, img_width = img.shape[:2]
            
            # Etiketi oku
            with open(label_path, 'r') as f:
                yolo_annotations = f.readlines()
            
            # Bounding box'ları dönüştür
            bboxes = []
            class_labels = []
            for ann in yolo_annotations:
                bbox = yolo_to_bbox(ann, img_width, img_height)
                if bbox is not None:
                    class_labels.append(bbox[0])
                    bboxes.append(bbox[1:])
            
            # Sınıfa özel dönüşüm seç
            transform = get_transform_by_class(class_id)
            
            # Her görüntü için birden fazla çoğaltma yap
            aug_count = 0
            for _ in range(augmentation_factor):
                # Temel çoğaltmalar (her bir görüntü için 2 kopya)
                for i in range(2):
                    try:
                        transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                        transformed_img = transformed['image']
                        transformed_bboxes = transformed['bboxes']
                        transformed_class_labels = transformed['class_labels']
                        
                        # Çoğaltılmış görüntüyü kaydet
                        aug_img_file = f"{os.path.splitext(img_file)[0]}_aug_{aug_count}.jpg"
                        aug_img_path = os.path.join(OUTPUT_IMAGES_DIR, aug_img_file)
                        cv2.imwrite(aug_img_path, cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR))
                        
                        # Çoğaltılmış etiketi kaydet
                        aug_label_file = f"{os.path.splitext(img_file)[0]}_aug_{aug_count}.txt"
                        aug_label_path = os.path.join(OUTPUT_LABELS_DIR, aug_label_file)
                        with open(aug_label_path, 'w') as f:
                            for bbox, cls_id in zip(transformed_bboxes, transformed_class_labels):
                                x_min, y_min, x_max, y_max = bbox
                                img_height, img_width = transformed_img.shape[:2]
                                yolo_line = bbox_to_yolo(cls_id, x_min, y_min, x_max, y_max, img_width, img_height)
                                f.write(yolo_line + '\n')
                        
                        aug_count += 1
                    except Exception as e:
                        print(f"Hata: {e}")

def apply_advanced_augmentation(class_images):
    """Mixup ve Mozaik gibi gelişmiş veri çoğaltma teknikleri uygular"""
    print("Mixup ve Mozaik çoğaltma uygulanıyor...")
    
    # Sadece görüntü sayısı sıfırdan büyük olan sınıflar için işlem yap
    active_classes = {class_id: info for class_id, info in CLASS_INFO.items() if info['count'] > 0}
    
    for class_id in [cid for cid in class_images.keys() if cid in active_classes]:
        if len(class_images[class_id]) < 2:
            continue
        
        # Gereken ekstra görüntü sayısı
        current_count = CLASS_INFO[class_id]["count"] * ((CLASS_INFO[class_id]["target"] // CLASS_INFO[class_id]["count"]) + 1) * 2
        extra_needed = max(0, CLASS_INFO[class_id]["target"] - current_count)
        
        if extra_needed <= 0:
            continue
        
        # Mixup işlemi
        print(f"Sınıf {class_id} için {extra_needed} ekstra görüntü oluşturuluyor...")
        for i in tqdm(range(min(extra_needed, 50)), desc=f"Sınıf {class_id} Mixup"):
            # İki rastgele görüntü seç
            img_files = random.sample(class_images[class_id], 2)
            img1_path = os.path.join(IMAGES_DIR, img_files[0])
            img2_path = os.path.join(IMAGES_DIR, img_files[1])
            
            label1_path = os.path.join(LABELS_DIR, os.path.splitext(img_files[0])[0] + '.txt')
            label2_path = os.path.join(LABELS_DIR, os.path.splitext(img_files[1])[0] + '.txt')
            
            # Görüntüleri oku
            img1 = cv2.imread(img1_path)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img1 = cv2.resize(img1, (640, 640))
            
            img2 = cv2.imread(img2_path)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img2 = cv2.resize(img2, (640, 640))
            
            # Etiketleri oku
            bboxes1 = []
            with open(label1_path, 'r') as f:
                for line in f:
                    bbox = yolo_to_bbox(line, 640, 640)
                    if bbox is not None:
                        bboxes1.append(bbox)
            
            bboxes2 = []
            with open(label2_path, 'r') as f:
                for line in f:
                    bbox = yolo_to_bbox(line, 640, 640)
                    if bbox is not None:
                        bboxes2.append(bbox)
            
            # Mixup uygula
            mixed_img, mixed_labels = create_mixup(img1, img2, bboxes1, bboxes2)
            
            # Kaydet
            mixup_img_file = f"mixup_{class_id}_{i}.jpg"
            mixup_img_path = os.path.join(OUTPUT_IMAGES_DIR, mixup_img_file)
            cv2.imwrite(mixup_img_path, cv2.cvtColor(mixed_img, cv2.COLOR_RGB2BGR))
            
            mixup_label_file = f"mixup_{class_id}_{i}.txt"
            mixup_label_path = os.path.join(OUTPUT_LABELS_DIR, mixup_label_file)
            with open(mixup_label_path, 'w') as f:
                for bbox in mixed_labels:
                    class_id, x_min, y_min, x_max, y_max = bbox
                    yolo_line = bbox_to_yolo(class_id, x_min, y_min, x_max, y_max, 640, 640)
                    f.write(yolo_line + '\n')
        
        # Mozaik işlemi
        if len(class_images[class_id]) >= 4:
            for i in tqdm(range(min(extra_needed, 30)), desc=f"Sınıf {class_id} Mozaik"):
                # 4 rastgele görüntü seç
                img_files = random.sample(class_images[class_id], 4)
                images = []
                all_bboxes = []
                
                for img_file in img_files:
                    img_path = os.path.join(IMAGES_DIR, img_file)
                    label_path = os.path.join(LABELS_DIR, os.path.splitext(img_file)[0] + '.txt')
                    
                    # Görüntüyü oku
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (640, 640))
                    images.append(img)
                    
                    # Etiketi oku
                    bboxes = []
                    with open(label_path, 'r') as f:
                        for line in f:
                            bbox = yolo_to_bbox(line, 640, 640)
                            if bbox is not None:
                                bboxes.append(bbox)
                    all_bboxes.append(bboxes)
                
                # Mozaik uygula
                mosaic_img, mosaic_labels = create_mosaic(images, all_bboxes)
                
                # Kaydet
                mosaic_img_file = f"mosaic_{class_id}_{i}.jpg"
                mosaic_img_path = os.path.join(OUTPUT_IMAGES_DIR, mosaic_img_file)
                cv2.imwrite(mosaic_img_path, cv2.cvtColor(mosaic_img, cv2.COLOR_RGB2BGR))
                
                mosaic_label_file = f"mosaic_{class_id}_{i}.txt"
                mosaic_label_path = os.path.join(OUTPUT_LABELS_DIR, mosaic_label_file)
                with open(mosaic_label_path, 'w') as f:
                    for bbox in mosaic_labels:
                        class_id, x_min, y_min, x_max, y_max = bbox
                        yolo_line = bbox_to_yolo(class_id, x_min, y_min, x_max, y_max, 640, 640)
                        f.write(yolo_line + '\n')

def analyze_augmented_dataset():
    """Çoğaltılmış veri setinin istatistiklerini hesaplar"""
    print("\nÇoğaltılmış veri seti analiz ediliyor...")
    # Sadece görüntü sayısı sıfırdan büyük olan sınıflar için sayaç oluştur
    class_counts = {class_id: 0 for class_id, info in CLASS_INFO.items() if info['count'] > 0}
    image_files = [f for f in os.listdir(OUTPUT_IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(OUTPUT_LABELS_DIR, label_file)
        
        if not os.path.exists(label_path):
            continue
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    if class_id in class_counts:
                        class_counts[class_id] += 1
    
    print("Çoğaltılmış veri seti istatistikleri:")
    for class_id, count in class_counts.items():
        print(f"Sınıf {class_id} ({CLASS_INFO[class_id]['name']}): {count} adet etiket")
    
    print(f"\nToplam görüntü sayısı: {len(image_files)}")
    
    # Görüntü sayısı grafiğini çiz
    plt.figure(figsize=(10, 6))
    classes = [CLASS_INFO[class_id]['name'] for class_id in class_counts.keys()]
    original_counts = [CLASS_INFO[class_id]['count'] for class_id in class_counts.keys()]
    augmented_counts = [class_counts[class_id] for class_id in class_counts.keys()]
    
    x = np.arange(len(classes))
    width = 0.35
    
    plt.bar(x - width/2, original_counts, width, label='Orijinal')
    plt.bar(x + width/2, augmented_counts, width, label='Çoğaltılmış')
    
    plt.xlabel('Sınıflar')
    plt.ylabel('Etiket Sayısı')
    plt.title('Veri Çoğaltma Sonuçları')
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Grafik dosyasını kaydet
    plt.savefig(os.path.join(OUTPUT_DIR, 'augmentation_results.png'))
    plt.close()
    
    return class_counts