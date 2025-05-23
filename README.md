
# 🌿 Besin Eksikliği Tespit Modeli - YOLOv11 Tabanlı

Bu proje, bitkilerde besin eksikliği tespiti için YOLOv11 tabanlı derin öğrenme modellerinin eğitiminde kullanılacak veri setlerini çoğaltma ve model eğitimi amacıyla geliştirilmiştir. Yetersiz veri ile yüksek doğruluk oranlarına ulaşmak için gelişmiş veri çoğaltma (data augmentation) teknikleri ve YOLOv11'in en son özellikleri kullanılmaktadır.

## 📌 Proje Hakkında

Bitkilerde besin eksikliği tespiti, tarımsal üretim ve verimlilik için kritik öneme sahiptir. Bu proje, özellikle üç temel besin eksikliği türünü tespit etmeye odaklanmaktadır:

- **Potasyum (K) Eksikliği**: Genellikle yaprak kenarlarında sararma ile karakterize edilir  
- **Azot (N) Eksikliği**: Genellikle yapraklarda genel bir sararma ile tanımlanır  
- **Potasyum ve Azot Eksikliği**: Her iki eksikliğin birleşik belirtileri
- **Fosfor (P) Eksikliği**: Koyu yeşil-mor yapraklar, yaşlı yapraklarda bronzlaşma
- **Kalsiyum (Ca) Eksikliği**: Genç yapraklarda büyüme noktası bozuklukları, yaprak kenarlarında nekroz
- **Magnezyum (Mg) Eksikliği**: Damarlar arası kloroz, yaprak kenarlarında yukarı kıvrılma
- **Kükürt (S) Eksikliği**: Genç yapraklarda açık yeşil renk, büyümede yavaşlama
- **Demir (Fe) Eksikliği**: Genç yapraklarda belirgin damarlar arası kloroz
- **Manganez (Mn) Eksikliği**: Damarlar arasında benekli kloroz, küçük nekrotik lekeler
- **Çinko (Zn) Eksikliği**: Yapraklarda kloroz, küçük yapraklar, rozet oluşumu
- **Bakır (Cu) Eksikliği**: Koyu yeşil yapraklar, bükülme ve kıvrılma
- **Bor (B) Eksikliği**: Büyüme noktasında ölüm, yaprak kalınlaşması
- **Molibden (Mo) Eksikliği**: Açık yeşil-sarı renk, yaprak kenarlarında yanıklar
- **Klor (Cl) Eksikliği**: Bronzlaşma, soluk sarı renk, yaprak küçülmesi
- **Nikel (Ni) Eksikliği**: Nekroz, kloroz ve büyüme noktasında ölüm

## 📋 Özellikler

- **İleri Düzey Veri Çoğaltma Teknikleri**: Sınıfa özel çoğaltma stratejileri  
- **Mixup ve Mozaik Çoğaltma**: Farklı görüntüleri kombinleyerek yeni eğitim verileri oluşturma  
- **Sınıf Dengesi Optimizasyonu**: Az örnekli sınıflar için daha yoğun çoğaltma  
- **Otomatik Eğitim/Doğrulama Bölümlemesi**: Stratified sampling ile dengeli veri dağılımı  
- **YOLO Eğitim Yapılandırması**: Otomatik YAML konfigürasyon dosyası oluşturma  
- **Görselleştirme Araçları**: Çoğaltma sonuçlarını analiz etmek için grafikler

## 🔧 Kurulum

```bash
# Repository'yi klonlayın
git clone https://github.com/KULLANICI_ADI/besin-eksikligi-tespit.git
cd besin-eksikligi-tespit

# Gereksinimleri yükleyin
pip install -r requirements.txt
```

## 📁 Veri Seti Yapısı

Proje, aşağıdaki yapıda bir veri seti beklemektedir:

```
dataset/
  └── original/
      ├── images/  # Orijinal görüntüler (.jpg, .png, .jpeg)
      └── labels/  # YOLO formatında etiketler (.txt)
```

Etiketler, standart YOLO formatında olmalıdır:

```
<class_id> <x_center> <y_center> <width> <height>
```

## 🚀 Kullanım

### 1. Veri Hazırlama ve Çoğaltma

```bash
python main.py
```

Bu komut aşağıdaki işlemleri gerçekleştirir:

- Orijinal görüntülerin kontrolü ve kopyalanması  
- Temel ve gelişmiş veri çoğaltma tekniklerinin uygulanması  
- Eğitim ve doğrulama setlerine bölünmesi  
- Çoğaltma sonuçlarının analizi ve görselleştirilmesi  
- YOLO eğitimi için YAML konfigürasyon dosyasının oluşturulması  

### 2. Model İndirme ve Eğitim

```bash
python train.py
```

Bu komut size aşağıdaki seçenekleri sunar:

1. **Model İndirme Seçenekleri**:
   - Tüm YOLOv11 modellerini indir (n, s, m)
   - Sadece seçilen modeli indir
   - Mevcut modelleri kullan

2. **Eğitim Stratejileri**:
   - Hızlı Eğitim (YOLOv11n - 100 epoch)
   - Standart Eğitim (YOLOv11s - 200 epoch)
   - Detaylı Eğitim (YOLOv11m - 300 epoch)
   - İki Aşamalı Eğitim (Ön eğitim + İnce ayar)

Eğitim sonuçları `runs/train/besin_eksikligi` klasöründe toplanır.

Veri çoğaltma tamamlandıktan sonra modeli eğitmek için:

#### Tek Aşamalı Eğitim

```bash
python train.py --img 640 --batch 16 --epochs 300 --data dataset/augmented/besin_eksikligi.yaml --weights yolo11l.pt
```

#### İki Aşamalı Eğitim (Önerilen)

**1. Aşama: Genel Eğitim**

```bash
python train.py --img 640 --batch 16 --epochs 200 --data dataset/augmented/besin_eksikligi.yaml \
    --weights yolo11l.pt --patience 50 --label-smoothing 0.1
```

**2. Aşama: İnce Ayar**

```bash
python train.py --img 640 --batch 8 --epochs 100 --data dataset/augmented/besin_eksikligi.yaml \
    --weights runs/train/exp/weights/best.pt --patience 50 --freeze 10 --lr0 0.0001
```

## 📊 Sonuçlar

Veri çoğaltma işlemi sonucunda aşağıdaki yapıda bir veri seti oluşturulacaktır:

```
dataset/
  ├── original/        # Orijinal veri seti
  └── augmented/       # Çoğaltılmış veri seti
      ├── images/
      │   ├── train/
      │   └── val/
      ├── labels/
      │   ├── train/
      │   └── val/
      ├── besin_eksikligi.yaml
      ├── augmentation_results.png
      └── class_*_augmentations.png
```

## 🧱 Sistem Mimarisi

Sistem modüler bir yapıda geliştirilmiştir:

- `main.py`: Veri çoğaltma ve hazırlama işlemlerini yöneten ana program
- `train.py`: YOLO11 model indirme, yükleme ve eğitim işlemlerini yöneten program
  * Ultralytics paket kontrolü ve yükleme
  * YOLO11 modellerini indirme (n, s, m, l)
  * 5 farklı eğitim stratejisi
  * GPU destekli eğitim
  * Eğitim sonuçlarının izlenmesi
- `config.py`: Yapılandırma ayarları  
- `utils.py`: Yardımcı fonksiyonlar  
- `transforms.py`: Veri çoğaltma dönüşümleri  
- `augmentation.py`: Veri çoğaltma işlemleri  
- `dataset.py`: Veri seti oluşturma işlemleri  
- `visualization.py`: Görselleştirme araçları  

## 📈 Performans

| Sınıf                        | Orijinal Görüntü | Çoğaltılmış |
|-----------------------------|------------------|-------------|
| Potasyum (K) Eksikliği          | 36               | ~250        |
| Azot (N) Eksikliği              | 47               | ~250        |
| Azot ve Potasyum Eksikliği      | 40               | ~250        |
| Fosfor (P) Eksikliği            | 0                | ~250        |
| Kalsiyum (Ca) Eksikliği         | 0                | ~250        |
| Magnezyum (Mg) Eksikliği        | 0                | ~250        |
| Kükürt (S) Eksikliği           | 0                | ~250        |
| Demir (Fe) Eksikliği            | 0                | ~250        |
| Manganez (Mn) Eksikliği         | 0                | ~250        |
| Çinko (Zn) Eksikliği            | 0                | ~250        |
| Bakır (Cu) Eksikliği           | 0                | ~250        |
| Bor (B) Eksikliği               | 0                | ~250        |
| Molibden (Mo) Eksikliği         | 0                | ~250        |
| Klor (Cl) Eksikliği             | 0                | ~250        |
| Nikel (Ni) Eksikliği            | 0                | ~250        |

## 📝 Konfigürasyon

`config.py` dosyasını düzenleyerek sistem parametrelerini değiştirebilirsiniz:

```python
CLASS_INFO = {
    0: {"name": "Potasyum Eksikliği", "count": 36, "target": 500},
    1: {"name": "Azot Eksikliği", "count": 47, "target": 500},
    2: {"name": "Azot ve Potasyum Eksikliği", "count": 40, "target": 500}
}

INPUT_DIR = "dataset/original"
OUTPUT_DIR = "dataset/augmented"
TEST_RATIO = 0.2
```

## 🔮 İleri Düzey Kullanım

### Özel Dönüşümler

#### Azot ve Potasyum Eksikliği için Özel Çoğaltma

```python
def get_combined_deficiency_transform():
    return A.Compose([
        # Azot eksikliği belirtileri
        A.ColorJitter(
            brightness=0.2,  # Genel sararma için parlaklık artışı
            contrast=0.2,    # Kontrast ayarı
            saturation=0.3,  # Doygunluk azaltma
            hue=0.1,        # Renk tonu değişimi
            p=0.8
        ),
        # Potasyum eksikliği belirtileri
        A.OneOf([
            # Yaprak kenarlarında sararma efekti
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.2,
                p=1.0
            ),
            # Nekroz benzeri koyu lekeler
            A.MultiplicativeNoise(
                multiplier=[0.7, 0.9],
                per_channel=True,
                p=1.0
            )
        ], p=0.8),
        # Genel doku ve şekil değişiklikleri
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.2, p=1.0),  # Yaprak kıvrılmaları
            A.GridDistortion(distort_limit=0.2, p=1.0),      # Doku bozulmaları
            A.ElasticTransform(alpha=120, sigma=120, p=1.0)  # Elastik deformasyonlar
        ], p=0.5),
        # Gerçekçilik artırıcı efektler
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),   # Doku detayları
            A.Sharpen(alpha=(0.2, 0.5), p=1.0),             # Kenar belirginleştirme
            A.Emboss(alpha=(0.2, 0.5), p=1.0)               # Kabartma efekti
        ], p=0.3)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
```

#### Diğer Özel Dönüşümler

```python
def get_custom_transform():
    return A.Compose([
        A.RandomSizedCrop(min_max_height=(400, 500), height=640, width=640, p=0.5),
        A.OpticalDistortion(distort_limit=0.2, shift_limit=0.05, p=0.3),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
```

### Test Zamanı Çoğaltma (TTA)

```python
def predict_with_tta(model, image_path, conf=0.25, iou=0.45):
    # Test zamanı çoğaltma ile tahmin
    img = cv2.imread(image_path)
    results = model(img, conf=conf, iou=iou)
    
    img_h_flip = cv2.flip(img, 1)
    results_h_flip = model(img_h_flip, conf=conf, iou=iou)
    
    final_results = ensemble_predictions([results, results_h_flip])
    
    return final_results
```

## 📄 Lisans

Bu proje [... Lisansı](LICENSE) altında lisanslanmıştır.

## 👥 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen bir pull request oluşturun veya bir issue açarak projeye katkıda bulunun.

## 📞 İletişim

Sorularınız için lütfen `[e-posta adresiniz]` adresine e-posta gönderin veya bir GitHub issue açın.

---

> Bu proje, tarımsal üretimde daha sürdürülebilir uygulamaları desteklemek ve bitki sağlığı izlemede yapay zeka teknolojilerinin kullanımını yaygınlaştırmak amacıyla geliştirilmiştir.
