import os
import subprocess

def check_and_install_ultralytics():
    """Ultralytics paketini kontrol et ve gerekirse yükle"""
    try:
        import ultralytics
        print("Ultralytics paketi zaten yüklü.")
    except ImportError:
        print("Ultralytics paketi yükleniyor...")
        subprocess.run(["pip", "install", "ultralytics"], check=True)
        print("Ultralytics paketi başarıyla yüklendi!")

# Ultralytics'i kontrol et ve yükle
check_and_install_ultralytics()
from ultralytics import YOLO

def download_yolo_models(choice='1'):
    """YOLO modellerini indir"""
    print("\nYOLO modellerini indirme işlemi başlıyor...")
    
    try:
        if choice == '1':
            print("Tüm modeller indiriliyor...")
            print("YOLO11n indiriliyor...")
            YOLO('yolo11n.pt')
            print("YOLO11s indiriliyor...")
            YOLO('yolo11s.pt')
            print("YOLO11m indiriliyor...")
            YOLO('yolo11m.pt')
            print("Tüm modeller indirildi!")
            return 'yolo11n.pt'  # Test eğitimi için en küçük modeli döndür
        elif choice == '2':
            models = {
                '1': 'yolo11n.pt',
                '2': 'yolo11s.pt',
                '3': 'yolo11m.pt',
                '4': 'yolo11l.pt'
            }
            print("\nHangi modeli indirmek istersiniz?")
            print("1. YOLO11n (küçük)")
            print("2. YOLO11s (orta)")
            print("3. YOLO11m (büyük)")
            print("4. YOLO11l (büyük)")
            model_choice = input("Seçiminiz (1/2/3/4): ")
            
            if model_choice in models:
                model_name = models[model_choice]
                print(f"\n{model_name} indiriliyor...")
                YOLO(model_name)
                print("Model indirildi!")
                return model_name
        else:
            print("Mevcut modeller kullanılacak.")
            return 'yolo11n.pt'
            
    except Exception as e:
        print(f"Model indirme hatası: {e}")
        return None

def train_yolo(model_name='yolov8n.pt', epochs=10, batch_size=16, image_size=640):
    """YOLO modelini eğit"""
    # Veri seti yolu kontrolü
    yaml_path = 'dataset/yolo/dataset.yaml'
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"dataset.yaml dosyası bulunamadı: {yaml_path}")

    print("\nYOLO eğitimi başlıyor...")
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {image_size}")

    try:
        model = YOLO(model_name)
        
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            device=0,  # GPU kullan
            project='runs/train',
            name='besin_eksikligi',
            exist_ok=True,
            pretrained=True,
            optimizer='Adam',
            verbose=True,
            seed=42,
            deterministic=True
        )
        
        print("\nEğitim tamamlandı!")
        print(f"Sonuçlar: runs/train/besin_eksikligi klasöründe")
        return True
        
    except Exception as e:
        print(f"Eğitim hatası: {e}")
        return False

def main():
    """Ana eğitim akışı"""
    print("\nYOLO modellerini indirmek ister misiniz?")
    print("1. Tüm modelleri indir (YOLOv8n, YOLOv8s, YOLOv8m)")
    print("2. Sadece seçilen modeli indir")
    print("3. Model indirme (zaten indirilmiş)")
    
    download_choice = input("\nSeçiminiz (1/2/3/4): ")
    model_name = download_yolo_models(download_choice)
    
    if model_name:
        print("\nEğitim için seçenekler:")
        print("1. Hızlı Eğitim (YOLO11n - 100 epoch)")
        print("2. Standart Eğitim (YOLO11s - 200 epoch)")
        print("3. Detaylı Eğitim (YOLO11m - 300 epoch)")
        print("4. Detaylı Eğitim (YOLO11l - 400 epoch)")
        print("5. İki Aşamalı Eğitim (YOLO11l - Ön eğitim (100 epochs) + İnce ayar(1000 epochs))")
        
        train_choice = input("\nHangi eğitim modelini kullanmak istersiniz? (1/2/3/4): ")
        
        if train_choice == '1':
            train_yolo(model_name=model_name, epochs=100, batch_size=16)
        elif train_choice == '2':
            train_yolo(model_name='yolo11s.pt', epochs=200, batch_size=16)
        elif train_choice == '3':
            train_yolo(model_name='yolo11m.pt', epochs=300, batch_size=16)
        elif train_choice == '4':
            train_yolo(model_name='yolo11l.pt', epochs=400, batch_size=16)
        elif train_choice == '5':
            # İki aşamalı eğitim
            print("\nÖn eğitim başlıyor...")
            train_yolo(model_name='yolo11l.pt', epochs=100, batch_size=16)
            print("\nİnce ayar eğitimi başlıyor...")
            train_yolo(model_name='runs/train/besin_eksikligi/weights/best.pt', epochs=1000, batch_size=32)

if __name__ == "__main__":
    main()
