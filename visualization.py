# visualization.py - Görselleştirme işlemleri

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from config import CLASS_INFO, IMAGES_DIR, LABELS_DIR, OUTPUT_DIR
from utils import yolo_to_bbox
from transforms import get_transform_by_class

def visualize_augmentations():
    """Çoğaltma örneklerini görselleştirir"""
    print("Çoğaltma örnekleri görselleştiriliyor...")
    
    # Her sınıf için bir orijinal resim seç
    class_sample_images = {}
    for class_id in CLASS_INFO.keys():
        class_images_list = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        for img_file in class_images_list:
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(LABELS_DIR, label_file)
            
            if not os.path.exists(label_path):
                continue
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5 and int(parts[0]) == class_id:
                        class_sample_images[class_id] = img_file
                        break
            
            if class_id in class_sample_images:
                break
    
    # Her sınıf için çoğaltma örneklerini görselleştir
    for class_id, img_file in class_sample_images.items():
        img_path = os.path.join(IMAGES_DIR, img_file)
        label_path = os.path.join(LABELS_DIR, os.path.splitext(img_file)[0] + '.txt')
        
        # Görüntüyü oku
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width = img.shape[:2]
        
        # Etiketi oku
        bboxes = []
        class_labels = []
        with open(label_path, 'r') as f:
            for line in f:
                bbox = yolo_to_bbox(line, img_width, img_height)
                if bbox is not None:
                    class_labels.append(bbox[0])
                    bboxes.append(bbox[1:])
        
        # Sınıfa özel çoğaltma seç
        transform = get_transform_by_class(class_id)
        
        # Sınıf adını belirle
        if class_id == 0:
            title = "Potasyum Eksikliği Çoğaltmaları"
        elif class_id == 1:
            title = "Azot Eksikliği Çoğaltmaları"
        elif class_id == 2:
            title = "Azot ve Potasyum Eksikliği Çoğaltmaları"
        else:
            title = "Temel Çoğaltmalar"
        
        # 9 farklı çoğaltma örneği oluştur
        plt.figure(figsize=(15, 15))
        plt.subplot(3, 3, 1)
        plt.imshow(img)
        plt.title("Orijinal")
        plt.axis('off')
        
        for i in range(8):
            try:
                transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                transformed_img = transformed['image']
                transformed_bboxes = transformed['bboxes']
                transformed_class_labels = transformed['class_labels']
                
                plt.subplot(3, 3, i+2)
                plt.imshow(transformed_img)
                plt.title(f"Çoğaltma {i+1}")
                plt.axis('off')
                
                # Bounding box'ları çiz
                for bbox, cls_id in zip(transformed_bboxes, transformed_class_labels):
                    x_min, y_min, x_max, y_max = [int(c) for c in bbox]
                    color = (0, 1, 0)  # Yeşil
                    plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                                                     fill=False, edgecolor=color, linewidth=2))
            except Exception as e:
                print(f"Görselleştirme hatası: {e}")
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"class_{class_id}_augmentations.png"))
        plt.close()