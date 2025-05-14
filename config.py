# config.py - Yapılandırma ayarları

import os

# Ana dizin
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Veri seti bilgileri
CLASS_INFO = {
    0: {"name": "Potasyum (K) Eksikliği", "count": 36, "target": 250},
    1: {"name": "Azot (N) Eksikliği", "count": 47, "target": 250},
    2: {"name": "Azot ve Potasyum Eksikliği", "count": 40, "target": 250},
    3: {"name": "Fosfor (P) Eksikliği", "count": 0, "target": 250},
    4: {"name": "Kalsiyum (Ca) Eksikliği", "count": 0, "target": 250},
    5: {"name": "Magnezyum (Mg) Eksikliği", "count": 0, "target": 250},
    6: {"name": "Kükürt (S) Eksikliği", "count": 0, "target": 250},
    7: {"name": "Demir (Fe) Eksikliği", "count": 0, "target": 250},
    8: {"name": "Manganez (Mn) Eksikliği", "count": 0, "target": 250},
    9: {"name": "Çinko (Zn) Eksikliği", "count": 0, "target": 250},
    10: {"name": "Bakır (Cu) Eksikliği", "count": 0, "target": 250},
    11: {"name": "Bor (B) Eksikliği", "count": 0, "target": 250},
    12: {"name": "Molibden (Mo) Eksikliği", "count": 0, "target": 250},
    13: {"name": "Klor (Cl) Eksikliği", "count": 0, "target": 250},
    14: {"name": "Nikel (Ni) Eksikliği", "count": 0, "target": 250}
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
