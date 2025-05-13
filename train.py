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

def train_yolo(model_name='yolo11n.pt', epochs=10, batch_size=16, image_size=640,
             lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
             warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
             save_period=50, patience=50, val_period=10):
    """YOLO modelini eğit
    Args:
        model_name (str): Model dosyası
        epochs (int): Toplam epoch sayısı
        batch_size (int): Batch boyutu
        image_size (int): Görüntü boyutu
        lr0 (float): Başlangıç learning rate
        lrf (float): Final learning rate (lr0 * lrf)
        momentum (float): SGD momentum/Adam beta1
        weight_decay (float): Optimizer weight decay
        warmup_epochs (float): Warmup epoch sayısı
        warmup_momentum (float): Warmup başlangıç momentum
        warmup_bias_lr (float): Warmup başlangıç bias lr
        save_period (int): Kaç epoch'ta bir kayıt alınacağı
    """
    # Veri seti yolu kontrolü
    yaml_path = 'dataset/yolo/dataset.yaml'
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"dataset.yaml dosyası bulunamadı: {yaml_path}")

    print("\nYOLO eğitimi başlıyor...")
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {image_size}")
    print(f"Learning Rate: {lr0}")
    print(f"Save Period: Her {save_period} epoch'ta bir")
    print(f"Early Stopping Patience: {patience} epoch")
    print(f"Validation Period: Her {val_period} epoch'ta bir")

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
            deterministic=True,
            resume=True,  # Eğitimi kaldığı yerden devam ettir
            save_period=save_period,  # Her N epoch'ta bir kaydet
            plots=True,  # Eğitim grafiklerini kaydet
            save_json=True,  # Detaylı metrikleri JSON olarak kaydet
            # Validation ve Early Stopping
            val=val_period,  # Her N epoch'ta bir validation yap
            patience=patience,  # Early stopping için beklenecek epoch sayısı
            # Hyperparametreler
            lr0=lr0,  # başlangıç learning rate
            lrf=lrf,  # final learning rate (lr0 * lrf)
            momentum=momentum,  # SGD momentum/Adam beta1
            weight_decay=weight_decay,  # optimizer weight decay
            warmup_epochs=warmup_epochs,  # warmup epochs
            warmup_momentum=warmup_momentum,  # warmup başlangıç momentum
            warmup_bias_lr=warmup_bias_lr  # warmup başlangıç bias lr
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
            train_yolo(model_name='runs/train/besin_eksikligi/weights/best.pt', epochs=500, batch_size=32)
            print("\nİnce ayar eğitimi başlıyor...")
            train_yolo(model_name='runs/train/besin_eksikligi/weights/best.pt', epochs=1000, batch_size=32)

if __name__ == "__main__":
    main()
