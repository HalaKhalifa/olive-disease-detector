import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(data_dir, image_size=(224, 224), batch_size=32):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    train_val_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.3,
        shear_range=0.2,
        brightness_range=(0.7, 1.3),
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_val_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset="training",
        shuffle=True,
        seed=42
    )

    val_gen = train_val_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset="validation",
        shuffle=False,
        seed=42
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )

    return train_gen, val_gen, test_gen
