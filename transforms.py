# transforms.py - Veri çoğaltma dönüşümleri

import albumentations as A

# Temel Dönüşüm - Tüm sınıflar için
def get_basic_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.Resize(height=640, width=640, p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Potasyum Eksikliği için özel çoğaltma
def get_potassium_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=20, p=0.8),
        # Potasyum eksikliği belirtilerini vurgulayan renk ayarları (yaprak kenarlarındaki sararmalar)
        A.OneOf([
            A.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.3, hue=0.05, p=0.7),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=20, p=0.5),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=-5, p=0.5),
        ], p=0.9),
        # Bitki dokusunu geliştiren özel ayarlar
        A.OneOf([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
        ], p=0.7),
        A.RandomCrop(height=480, width=480, p=0.3),
        A.Resize(height=640, width=640, p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Azot Eksikliği için özel çoğaltma
def get_nitrogen_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=20, p=0.8),
        # Azot eksikliği belirtilerini vurgulayan renk ayarları (genel sararmalar)
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=20, p=0.7),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.4, hue=0.05, p=0.5),
            A.RGBShift(r_shift_limit=5, g_shift_limit=15, b_shift_limit=5, p=0.5),
        ], p=0.9),
        # Bitki dokusunu geliştiren özel ayarlar
        A.OneOf([
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
            A.Sharpen(alpha=(0.2, 0.4), lightness=(0.6, 1.0), p=0.5),
        ], p=0.7),
        A.RandomCrop(height=480, width=480, p=0.3),
        A.Resize(height=640, width=640, p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Azot ve Potasyum Eksikliği için özel çoğaltma
def get_both_deficiency_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=20, p=0.8),
        # Karma eksiklik belirtilerini vurgulayan renk ayarları
        A.OneOf([
            A.ColorJitter(brightness=0.1, contrast=0.15, saturation=0.35, hue=0.05, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=20, p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=10, b_shift_limit=-5, p=0.5),
        ], p=0.9),
        # Bitki dokusunu geliştiren özel ayarlar
        A.OneOf([
            A.CLAHE(clip_limit=3.5, tile_grid_size=(8, 8), p=0.5),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
        ], p=0.7),
        A.RandomCrop(height=480, width=480, p=0.3),
        A.Resize(height=640, width=640, p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Fosfor Eksikliği için özel çoğaltma
def get_phosphorus_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=20, p=0.8),
        # Fosfor eksikliği belirtileri: Koyu yeşil-mor yapraklar, yaşlı yapraklarda bronzlaşma
        A.OneOf([
            A.ColorJitter(brightness=-0.2, contrast=0.3, saturation=0.4, hue=0.15, p=0.7),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=-30, p=0.6),
            A.RGBShift(r_shift_limit=20, g_shift_limit=-10, b_shift_limit=25, p=0.5),
        ], p=0.9),
        # Doku değişiklikleri
        A.OneOf([
            A.Sharpen(alpha=(0.3, 0.7), lightness=(0.5, 1.0), p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        ], p=0.7),
        A.Resize(height=640, width=640, p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Kalsiyum Eksikliği için özel çoğaltma
def get_calcium_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=20, p=0.8),
        # Kalsiyum eksikliği belirtilerini vurgulayan renk ayarları
        A.OneOf([
            A.ColorJitter(brightness=0.15, contrast=0.25, saturation=0.2, hue=0.05, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=25, p=0.6),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=5, p=0.5),
        ], p=0.9),
        A.Resize(height=640, width=640, p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Magnezyum Eksikliği için özel çoğaltma
def get_magnesium_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=20, p=0.8),
        # Magnezyum eksikliği belirtileri: Damarlar arası kloroz, yaprak kenarlarında yukarı kıvrılma
        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.25, saturation=0.4, hue=0.1, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=40, val_shift_limit=30, p=0.6),
            A.RGBShift(r_shift_limit=15, g_shift_limit=25, b_shift_limit=-10, p=0.5),
        ], p=0.9),
        # Doku ve kontrast ayarları
        A.OneOf([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.6),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.6, 1.0), p=0.4),
        ], p=0.7),
        A.Resize(height=640, width=640, p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Kükürt Eksikliği için özel çoğaltma
def get_sulfur_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=20, p=0.8),
        # Kükürt eksikliği belirtilerini vurgulayan renk ayarları
        A.OneOf([
            A.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.25, hue=0.05, p=0.7),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=25, val_shift_limit=20, p=0.6),
            A.RGBShift(r_shift_limit=8, g_shift_limit=12, b_shift_limit=5, p=0.5),
        ], p=0.9),
        A.Resize(height=640, width=640, p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Demir Eksikliği için özel çoğaltma
def get_iron_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=20, p=0.8),
        # Demir eksikliği belirtileri: Genç yapraklarda belirgin damarlar arası kloroz
        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.1, p=0.7),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=45, val_shift_limit=35, p=0.6),
            A.RGBShift(r_shift_limit=20, g_shift_limit=25, b_shift_limit=-15, p=0.5),
        ], p=0.9),
        # Damar kontrastını artırma
        A.OneOf([
            A.CLAHE(clip_limit=5.0, tile_grid_size=(8, 8), p=0.6),
            A.Sharpen(alpha=(0.3, 0.6), lightness=(0.7, 1.0), p=0.4),
        ], p=0.8),
        A.Resize(height=640, width=640, p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Manganez Eksikliği için özel çoğaltma
def get_manganese_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=20, p=0.8),
        # Manganez eksikliği belirtilerini vurgulayan renk ayarları
        A.OneOf([
            A.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.3, hue=0.05, p=0.7),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=25, val_shift_limit=20, p=0.6),
            A.RGBShift(r_shift_limit=10, g_shift_limit=12, b_shift_limit=5, p=0.5),
        ], p=0.9),
        A.Resize(height=640, width=640, p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Çinko Eksikliği için özel çoğaltma
def get_zinc_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=20, p=0.8),
        # Çinko eksikliği belirtileri: Yapraklarda kloroz, küçük yapraklar, rozet oluşumu
        A.OneOf([
            A.ColorJitter(brightness=0.25, contrast=0.3, saturation=0.35, hue=0.1, p=0.7),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=35, val_shift_limit=25, p=0.6),
            A.RGBShift(r_shift_limit=15, g_shift_limit=20, b_shift_limit=-5, p=0.5),
        ], p=0.9),
        # Doku ve boyut değişiklikleri
        A.OneOf([
            A.RandomScale(scale_limit=(-0.2, 0.1), p=0.4),  # Küçük yapraklar için
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.6),
        ], p=0.7),
        A.Resize(height=640, width=640, p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Bakır Eksikliği için özel çoğaltma
def get_copper_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=20, p=0.8),
        # Bakır eksikliği belirtileri: Koyu yeşil yapraklar, bükülme ve kıvrılma
        A.OneOf([
            A.ColorJitter(brightness=-0.2, contrast=0.3, saturation=0.4, hue=0.1, p=0.7),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=35, val_shift_limit=-25, p=0.6),
            A.RGBShift(r_shift_limit=-10, g_shift_limit=20, b_shift_limit=5, p=0.5),
        ], p=0.9),
        # Doku ve şekil değişiklikleri
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),  # Kıvrılma efekti
            A.OpticalDistortion(distort_limit=0.2, shift_limit=0.15, p=0.5),  # Bükülme efekti
        ], p=0.7),
        A.Resize(height=640, width=640, p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Bor Eksikliği için özel çoğaltma
def get_boron_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=20, p=0.8),
        # Bor eksikliği belirtileri: Büyüme noktasında ölüm, yaprak kalınlaşması
        A.OneOf([
            A.ColorJitter(brightness=-0.1, contrast=0.3, saturation=0.2, hue=0.05, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=-20, p=0.6),
            A.RGBShift(r_shift_limit=10, g_shift_limit=-5, b_shift_limit=5, p=0.5),
        ], p=0.9),
        # Doku ve kalınlık efektleri
        A.OneOf([
            A.Emboss(alpha=(0.2, 0.5), strength=(0.5, 1.0), p=0.5),  # Kalınlaşma efekti
            A.Sharpen(alpha=(0.3, 0.6), lightness=(0.5, 0.8), p=0.5),
        ], p=0.7),
        A.Resize(height=640, width=640, p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Molibden Eksikliği için özel çoğaltma
def get_molybdenum_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=20, p=0.8),
        # Molibden eksikliği belirtileri: Açık yeşil-sarı renk, yaprak kenarlarında yanıklar
        A.OneOf([
            A.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.3, hue=0.1, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=35, val_shift_limit=30, p=0.6),
            A.RGBShift(r_shift_limit=15, g_shift_limit=20, b_shift_limit=-10, p=0.5),
        ], p=0.9),
        # Kenar yanıkları için efektler
        A.OneOf([
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),  # Yanık efekti
            A.Sharpen(alpha=(0.3, 0.6), lightness=(0.6, 0.9), p=0.5),
        ], p=0.7),
        A.Resize(height=640, width=640, p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Klor Eksikliği için özel çoğaltma
def get_chlorine_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=20, p=0.8),
        # Klor eksikliği belirtileri: Bronzlaşma, soluk sarı renk, yaprak küçülmesi
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.15, p=0.7),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=-20, p=0.6),
            A.RGBShift(r_shift_limit=25, g_shift_limit=15, b_shift_limit=-10, p=0.5),
        ], p=0.9),
        # Boyut ve doku değişiklikleri
        A.OneOf([
            A.RandomScale(scale_limit=(-0.3, 0.1), p=0.4),  # Küçülme efekti
            A.Emboss(alpha=(0.2, 0.5), strength=(0.4, 0.8), p=0.6),  # Bronzlaşma efekti
        ], p=0.7),
        A.Resize(height=640, width=640, p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Nikel Eksikliği için özel çoğaltma
def get_nickel_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=20, p=0.8),
        # Nikel eksikliği belirtileri: Nekroz, kloroz ve büyüme noktasında ölüm
        A.OneOf([
            A.ColorJitter(brightness=-0.1, contrast=0.35, saturation=0.2, hue=0.1, p=0.7),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=-25, p=0.6),
            A.RGBShift(r_shift_limit=20, g_shift_limit=-10, b_shift_limit=5, p=0.5),
        ], p=0.9),
        # Nekroz ve ölü doku efektleri
        A.OneOf([
            A.CoarseDropout(max_holes=12, max_height=24, max_width=24, p=0.5),  # Nekroz efekti
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),  # Doku bozulması
        ], p=0.7),
        A.Resize(height=640, width=640, p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Sınıfa göre dönüşüm getir
def get_transform_by_class(class_id):
    if class_id == 0:
        return get_potassium_transform()
    elif class_id == 1:
        return get_nitrogen_transform()
    elif class_id == 2:
        return get_both_deficiency_transform()
    elif class_id == 3:
        return get_phosphorus_transform()
    elif class_id == 4:
        return get_calcium_transform()
    elif class_id == 5:
        return get_magnesium_transform()
    elif class_id == 6:
        return get_sulfur_transform()
    elif class_id == 7:
        return get_iron_transform()
    elif class_id == 8:
        return get_manganese_transform()
    elif class_id == 9:
        return get_zinc_transform()
    elif class_id == 10:
        return get_copper_transform()
    elif class_id == 11:
        return get_boron_transform()
    elif class_id == 12:
        return get_molybdenum_transform()
    elif class_id == 13:
        return get_chlorine_transform()
    elif class_id == 14:
        return get_nickel_transform()
    else:
        return get_basic_transform()