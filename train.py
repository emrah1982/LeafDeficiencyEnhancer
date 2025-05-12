from ultralytics import YOLO
import os

def train_yolo(model_name='yolov8n.pt', epochs=10, batch_size=16, image_size=640):
    print(f"\nEğitim başlıyor...")
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {image_size}\n")

    # Veri seti yolu kontrolü
    yaml_path = 'dataset/yolo/dataset.yaml'
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"dataset.yaml dosyası bulunamadı: {yaml_path}")
    # Model yükleme
    try:
        model = YOLO(model_name)
        print("Model başarıyla yüklendi.")
    except Exception as e:
        raise Exception(f"Model yüklenirken hata: {e}")

    # Eğitim
    try:
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            name='besin_eksikligi_test_run'
        )
        print("\nEğitim başarıyla tamamlandı!")
        print(f"Sonuçlar: runs/detect/train/besin_eksikligi_test_run klasöründe")
    except Exception as e:
        raise Exception(f"Eğitim sırasında hata: {e}")

if __name__ == "__main__":
    train_yolo()
