from ultralytics import YOLO
import os

# Veri seti yolunu kontrol et
yaml_path = '/content/LeafDeficiencyEnhancer/dataset/yolo/dataset.yaml'
if not os.path.exists(yaml_path):
    raise FileNotFoundError(f"dataset.yaml dosyası bulunamadı: {yaml_path}")

# Model yükleme ve eğitim
print("\nYOLO eğitimi başlıyor...")
model = YOLO('yolov8n.pt')

# Eğitim parametreleri
results = model.train(
    data=yaml_path,
    epochs=10,
    imgsz=640,
    batch=16,
    device=0,  # GPU kullan
    project='/content/LeafDeficiencyEnhancer/runs/train',
    name='besin_eksikligi_test_run',
    exist_ok=True,
    pretrained=True,
    optimizer='Adam',
    verbose=True,
    seed=42,
    deterministic=True
)

print("\nEğitim tamamlandı!")
print(f"Sonuçlar: /content/LeafDeficiencyEnhancer/runs/train/besin_eksikligi_test_run klasöründe")
