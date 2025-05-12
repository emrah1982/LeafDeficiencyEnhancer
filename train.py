from ultralytics import YOLO

# Model
model = YOLO('yolov8n.pt')  # yüklü bir YOLOv8n modeli yükle

# Eğitim
results = model.train(
    data='dataset/yolo/dataset.yaml',  # veri seti yapılandırma dosyası
    epochs=100,                        # epoch sayısı
    imgsz=640,                        # görüntü boyutu
    batch=16,                         # batch size
    name='besin_eksikligi',           # deney adı
    patience=20,                      # early stopping patience
    save=True,                        # en iyi modeli kaydet
    device='0',                       # GPU kullan (eğer varsa)
    project='runs/train',             # proje dizini
    optimizer='Adam',                 # optimizer
    lr0=0.001,                       # başlangıç learning rate
    lrf=0.01,                        # final learning rate factor
    momentum=0.937,                  # SGD momentum/Adam beta1
    weight_decay=0.0005,             # optimizer weight decay
    warmup_epochs=3.0,               # warmup epochs
    warmup_momentum=0.8,             # warmup başlangıç momentum
    warmup_bias_lr=0.1,             # warmup başlangıç bias lr
    box=7.5,                         # box loss gain
    cls=0.5,                         # cls loss gain
    dfl=1.5,                         # dfl loss gain
    label_smoothing=0.1,             # label smoothing epsilon
    plots=True,                      # training plots
    save_period=10                   # her 10 epochta bir kaydet
)

# Validasyon
results = model.val()

# Test
results = model.predict('path/to/test/image.jpg', save=True, conf=0.5)
