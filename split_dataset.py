import os
import shutil
import random
from pathlib import Path

# Sabit değişkenler
DATASET_PATH = "dataset/augmented"
SPLIT_PATH = "dataset/yolo"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

def create_folders():
    """YOLO veri seti için gerekli klasörleri oluşturur"""
    splits = ['train', 'val', 'test']
    for split in splits:
        for folder in ['images', 'labels']:
            os.makedirs(os.path.join(SPLIT_PATH, split, folder), exist_ok=True)

def split_dataset():
    """Veri setini train, validation ve test olarak böler"""
    print("\nVeri seti train, validation ve test olarak bölünüyor...")
    
    # Görüntü dosyalarını listele
    images_path = os.path.join(DATASET_PATH, "images")
    image_files = []
    
    # Önce train ve val klasörlerinden görüntüleri al
    for folder in ['train', 'val']:
        folder_path = os.path.join(images_path, folder)
        if os.path.exists(folder_path):
            files = [os.path.join(folder, f) for f in os.listdir(folder_path) 
                    if f.endswith(('.jpg', '.png', '.jpeg'))]
            image_files.extend(files)
    
    # Eğer görüntü bulunamazsa, doğrudan images klasörüne bak
    if not image_files and os.path.exists(images_path):
        files = [f for f in os.listdir(images_path) 
                if f.endswith(('.jpg', '.png', '.jpeg'))]
        image_files.extend(files)
    
    # Rastgele karıştır
    random.seed(RANDOM_SEED)
    random.shuffle(image_files)
    
    # Bölme noktalarını hesapla
    total_images = len(image_files)
    train_end = int(total_images * TRAIN_RATIO)
    val_end = train_end + int(total_images * VAL_RATIO)
    
    # Dosyaları böl
    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }
    
    # Her split için dosyaları kopyala
    for split_name, files in splits.items():
        print(f"\n{split_name.capitalize()} setine dosyalar kopyalanıyor...")
        for img_file in files:
            try:
                # Görüntü dosyasını kopyala
                if '/' in img_file or '\\' in img_file:
                    src_folder, img_name = os.path.split(img_file)
                    src_img = os.path.join(DATASET_PATH, "images", img_file)
                    src_label = os.path.join(DATASET_PATH, "labels", src_folder, os.path.splitext(img_name)[0] + '.txt')
                else:
                    img_name = img_file
                    src_img = os.path.join(DATASET_PATH, "images", img_name)
                    src_label = os.path.join(DATASET_PATH, "labels", os.path.splitext(img_name)[0] + '.txt')
                
                dst_img = os.path.join(SPLIT_PATH, split_name, "images", img_name)
                dst_label = os.path.join(SPLIT_PATH, split_name, "labels", os.path.splitext(img_name)[0] + '.txt')
                
                # Dosyaları kopyala
                shutil.copy2(src_img, dst_img)
                shutil.copy2(src_label, dst_label)
            except Exception as e:
                print(f"Hata: {e} - Dosya: {img_file}")
        
        print(f"{split_name.capitalize()} set: {len(files)} görüntü")

def create_yaml():
    """YOLO için dataset.yaml dosyası oluşturur"""
    yaml_content = f"""path: {os.path.abspath(SPLIT_PATH)}
train: train/images
val: val/images
test: test/images

# Sınıflar
names:
  0: Potasyum_Eksikligi
  1: Azot_Eksikligi
  2: Azot_ve_Potasyum_Eksikligi"""
    
    yaml_path = os.path.join(SPLIT_PATH, "dataset.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"\nYAML dosyası oluşturuldu: {yaml_path}")

def main():
    print("="*80)
    print("YOLO Veri Seti Hazırlama")
    print("="*80)
    
    # Klasörleri oluştur
    create_folders()
    
    # Veri setini böl
    split_dataset()
    
    # YAML dosyası oluştur
    create_yaml()
    
    print("\nVeri seti hazırlama tamamlandı!")
    print("="*80)
    print("\nEğitim için aşağıdaki komutu kullanabilirsiniz:")
    print("yolo task=detect mode=train model=yolov8n.pt data=dataset/yolo/dataset.yaml epochs=100 imgsz=640")

if __name__ == "__main__":
    main()
