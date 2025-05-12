# utils.py - Yardımcı fonksiyonlar

import os
import cv2
import numpy as np
import random
from config import CLASS_INFO

# Rastgele sayı üretici ayarı
random.seed(42)
np.random.seed(42)

# YOLO formatı dönüşüm fonksiyonları
def yolo_to_bbox(yolo_annotation, image_width, image_height):
    """YOLO formatındaki etiketi (class_id, x_center, y_center, width, height) 
    bounding box formatına (class_id, x_min, y_min, x_max, y_max) dönüştürür"""
    parts = yolo_annotation.strip().split()
    if len(parts) == 5:
        class_id, x_center, y_center, width, height = map(float, parts)
        x_min = (x_center - width/2) * image_width
        y_min = (y_center - height/2) * image_height
        x_max = (x_center + width/2) * image_width
        y_max = (y_center + height/2) * image_height
        return int(class_id), x_min, y_min, x_max, y_max
    return None

def bbox_to_yolo(class_id, x_min, y_min, x_max, y_max, image_width, image_height):
    """Bounding box formatını (class_id, x_min, y_min, x_max, y_max) 
    YOLO formatına (class_id, x_center, y_center, width, height) dönüştürür"""
    x_center = (x_min + x_max) / (2 * image_width)
    y_center = (y_min + y_max) / (2 * image_height)
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return f"{int(class_id)} {x_center} {y_center} {width} {height}"

# Görüntü ve etiketleri sınıfa göre organize et
def get_class_images(images_dir, labels_dir):
    """Her sınıf için görüntüleri belirler"""
    class_images = {0: [], 1: [], 2: []}
    
    # Görüntü klasörünü kontrol et
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'))]
    
    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        if not os.path.exists(label_path):
            continue
            
        # Dosya adından sınıfı belirle
        if '__K' in img_file and not '__N' in img_file:
            class_id = 0  # Potasyum Eksikliği
        elif '__N' in img_file and not '__K' in img_file:
            class_id = 1  # Azot Eksikliği
        elif '__N_K' in img_file or ('__N' in img_file and '__K' in img_file):
            class_id = 2  # Azot ve Potasyum Eksikliği
        else:
            continue
            
        class_images[class_id].append(img_file)
        CLASS_INFO[class_id]['count'] = len(class_images[class_id])
    
    # Sınıf sayılarını güncelle
    for class_id, images in class_images.items():
        print(f"Sınıf {class_id} ({CLASS_INFO[class_id]['name']}): {len(images)} görüntü")
    
    return class_images

# Mixup işlemi (iki görüntüyü birleştirme)
def create_mixup(img1, img2, labels1, labels2, alpha=0.5):
    """İki görüntüyü karma bir şekilde birleştirir"""
    # Rastgele karıştırma katsayısı
    lam = np.random.beta(alpha, alpha)
    
    # Görüntüleri karıştır
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mixed_img = cv2.addWeighted(img1, lam, img2, 1-lam, 0)
    mixed_img = mixed_img.astype(np.uint8)
    
    # Etiketleri birleştir - her iki görüntünün de etiketlerini ekle
    mixed_labels = []
    for label in labels1:
        if lam > 0.4:  # Birinci görüntünün katkısı yeterliyse
            mixed_labels.append(label)
    
    for label in labels2:
        if (1-lam) > 0.4:  # İkinci görüntünün katkısı yeterliyse
            mixed_labels.append(label)
    
    return mixed_img, mixed_labels

# Mozaik işlemi (4 görüntüyü birleştirme)
def create_mosaic(images, labels_list, output_size=640):
    """4 görüntüyü birleştirerek mozaik oluşturur"""
    mosaic_img = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    
    # Rastgele bir merkez noktası seç
    cx = int(random.uniform(output_size // 4, output_size * 3 // 4))
    cy = int(random.uniform(output_size // 4, output_size * 3 // 4))
    
    mosaic_labels = []
    
    # Her bir görüntüyü mozaiğin bir köşesine yerleştir
    for i, (img, img_labels) in enumerate(zip(images, labels_list)):
        h, w = img.shape[:2]
        
        # Mozaiğin farklı bir köşesi için başlangıç koordinatları
        if i == 0:  # Sol üst
            x1a, y1a, x2a, y2a = 0, 0, cx, cy
            x1b, y1b, x2b, y2b = w - cx, h - cy, w, h
        elif i == 1:  # Sağ üst
            x1a, y1a, x2a, y2a = cx, 0, output_size, cy
            x1b, y1b, x2b, y2b = 0, h - cy, w - cx, h
        elif i == 2:  # Sol alt
            x1a, y1a, x2a, y2a = 0, cy, cx, output_size
            x1b, y1b, x2b, y2b = w - cx, 0, w, h - cy
        elif i == 3:  # Sağ alt
            x1a, y1a, x2a, y2a = cx, cy, output_size, output_size
            x1b, y1b, x2b, y2b = 0, 0, w - cx, h - cy
        
        # Görüntüyü yerleştir
        mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        
        # Etiketleri ayarla
        for label in img_labels:
            class_id, x_min, y_min, x_max, y_max = label
            
            # Etiketleri yeni konuma taşı
            if i == 0:  # Sol üst
                x_min_new = x_min - w + cx + x1a
                y_min_new = y_min - h + cy + y1a
                x_max_new = x_max - w + cx + x1a
                y_max_new = y_max - h + cy + y1a
            elif i == 1:  # Sağ üst
                x_min_new = x_min + x1a
                y_min_new = y_min - h + cy + y1a
                x_max_new = x_max + x1a
                y_max_new = y_max - h + cy + y1a
            elif i == 2:  # Sol alt
                x_min_new = x_min - w + cx + x1a
                y_min_new = y_min + y1a
                x_max_new = x_max - w + cx + x1a
                y_max_new = y_max + y1a
            elif i == 3:  # Sağ alt
                x_min_new = x_min + x1a
                y_min_new = y_min + y1a
                x_max_new = x_max + x1a
                y_max_new = y_max + y1a
            
            # Sınırları kontrol et
            x_min_new = max(0, min(output_size-1, x_min_new))
            y_min_new = max(0, min(output_size-1, y_min_new))
            x_max_new = max(0, min(output_size-1, x_max_new))
            y_max_new = max(0, min(output_size-1, y_max_new))
            
            # Geçerli boyut kontrolü
            if x_max_new > x_min_new and y_max_new > y_min_new and (x_max_new-x_min_new)*(y_max_new-y_min_new) > 100:
                mosaic_labels.append([class_id, x_min_new, y_min_new, x_max_new, y_max_new])
    
    return mosaic_img, mosaic_labels