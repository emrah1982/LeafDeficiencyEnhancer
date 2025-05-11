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

# Sınıfa göre dönüşüm getir
def get_transform_by_class(class_id):
    if class_id == 0:
        return get_potassium_transform()
    elif class_id == 1:
        return get_nitrogen_transform()
    elif class_id == 2:
        return get_both_deficiency_transform()
    else:
        return get_basic_transform()