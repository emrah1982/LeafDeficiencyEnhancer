# config.py - Yapılandırma ayarları

import os

# Ana dizin
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Veri seti bilgileri
CLASS_INFO = {
    0: {"name": "Potasyum Eksikliği", "count": 41, "target": 250},
    1: {"name": "Azot Eksikliği", "count": 41, "target": 250},
    2: {"name": "Azot ve Potasyum Eksikliği", "count": 41, "target": 250}
}

# Klasör yapısı
#INPUT_DIR = "dataset/original"  # Orijinal veri seti
#OUTPUT_DIR = "dataset/augmented"  # Çoğaltılmış veri seti

INPUT_DIR = 'dataset/original'
OUTPUT_DIR = 'dataset/augmented'
IMAGES_DIR = os.path.join(INPUT_DIR, "images")
LABELS_DIR = os.path.join(INPUT_DIR, "labels")
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")

# Görselleştirme
VISUALIZE_EXAMPLES = True

# Eğitim parametreleri
TEST_RATIO = 0.2
RANDOM_SEED = 42
