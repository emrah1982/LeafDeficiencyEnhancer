# dataset.py - Veri seti oluşturma ve yönetme işlemleri

import os
import shutil
import random
from tqdm import tqdm
from config import OUTPUT_DIR, OUTPUT_IMAGES_DIR, OUTPUT_LABELS_DIR, TEST_RATIO

def split_dataset():
    """Veri setini eğitim ve doğrulama setlerine böler"""
    print("Veri seti eğitim ve doğrulama setlerine bölünüyor...")
    
    # Eğitim ve doğrulama klasörleri oluştur
    train_img_dir = os.path.join(OUTPUT_DIR, 'images/train')
    val_img_dir = os.path.join(OUTPUT_DIR, 'images/val')
    train_label_dir = os.path.join(OUTPUT_DIR, 'labels/train')
    val_label_dir = os.path.join(OUTPUT_DIR, 'labels/val')
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    # Her sınıf için resim dosyalarını sınıflandır
    class_images = {0: [], 1: [], 2: []}
    image_files = [f for f in os.listdir(OUTPUT_IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(OUTPUT_LABELS_DIR, label_file)
        
        if not os.path.exists(label_path):
            continue
        
        # Etiketi oku ve sınıfları bul
        with open(label_path, 'r') as f:
            lines = f.readlines()
            classes_in_image = set()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    classes_in_image.add(class_id)
        
        # Görüntüyü her içerdiği sınıfa ekle
        for class_id in classes_in_image:
            if class_id in class_images:
                class_images[class_id].append(img_file)
    
    # Her sınıf için ayrı ayrı eğitim/doğrulama bölünmesi yap
    val_images = set()
    
    for class_id, images in class_images.items():
        # Stratified split - her sınıf için aynı oranda doğrulama seti
        random.shuffle(images)  # Rastgele karıştır
        num_val = max(1, int(len(images) * TEST_RATIO))
        val_subset = set(images[:num_val])
        val_images.update(val_subset)
    
    # Dosyaları eğitim ve doğrulama klasörlerine taşı
    for img_file in tqdm(image_files, desc="Dosyalar bölünüyor"):
        label_file = os.path.splitext(img_file)[0] + '.txt'
        img_path = os.path.join(OUTPUT_IMAGES_DIR, img_file)
        label_path = os.path.join(OUTPUT_LABELS_DIR, label_file)
        
        if not os.path.exists(label_path):
            continue
        
        # Hedef yolları belirle
        if img_file in val_images:
            target_img_path = os.path.join(val_img_dir, img_file)
            target_label_path = os.path.join(val_label_dir, label_file)
        else:
            target_img_path = os.path.join(train_img_dir, img_file)
            target_label_path = os.path.join(train_label_dir, label_file)
        
        # Dosyaları kopyala
        shutil.copy(img_path, target_img_path)
        shutil.copy(label_path, target_label_path)
    
    # Eğitim ve doğrulama setlerinin istatistiklerini hesapla
    train_count = len(os.listdir(train_img_dir))
    val_count = len(os.listdir(val_img_dir))
    
    print(f"Eğitim seti: {train_count} görüntü")
    print(f"Doğrulama seti: {val_count} görüntü")
    print(f"Toplam: {train_count + val_count} görüntü")
    print(f"Doğrulama oranı: {val_count / (train_count + val_count):.2f}")

def create_yaml_file():
    """YOLO eğitimi için YAML dosyası oluşturur"""
    yaml_content = """
# YOLOv8/YOLO11 için veri seti konfigürasyonu
path: {path}  # Veri setinin kök dizini
train: images/train  # Eğitim görüntüleri
val: images/val  # Doğrulama görüntüleri

nc: 3  # Sınıf sayısı
names: 
  0: Potasyum Eksikliği
  1: Azot Eksikliği
  2: Azot ve Potasyum Eksikliği

# Sınıf ağırlıkları - dengesiz veri setleri için önemli
class_weights: [1.0, 1.0, 1.0]

# Hiperparametreler - besin eksikliği tespiti için optimize edilmiş
hyp:
  lr0: 0.005  # Başlangıç öğrenme oranı
  lrf: 0.0001  # Son öğrenme oranı
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 5.0
  warmup_momentum: 0.8
  warmup_bias_lr: 0.1
  box: 7.5  # Box loss ağırlığı
  cls: 1.0  # Class loss ağırlığı
  hsv_h: 0.015  # Hue Augmentation
  hsv_s: 0.7    # Saturation augmentation
  hsv_v: 0.4    # Value augmentation
  translate: 0.1  # Image translation
  scale: 0.7  # Image scale
  fliplr: 0.5  # Image flip left-right
  mosaic: 1.0  # Mozaik augmentation
  mixup: 0.15  # Mixup augmentation
""".format(path=os.path.abspath(OUTPUT_DIR))
    
    # YAML dosyasını kaydet
    yaml_path = os.path.join(OUTPUT_DIR, 'besin_eksikligi.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"YAML dosyası oluşturuldu: {yaml_path}")
    return yaml_path
