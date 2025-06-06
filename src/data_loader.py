from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(data_dir, image_size=(224, 224), batch_size=32, val_split=0.15):
    """
    Returns train, validation, and test data generators.

    Parameters:
        data_dir (str): Path to the dataset root directory containing 'train' and 'test' subfolders
        image_size (tuple): Target size for images
        batch_size (int): Batch size
        val_split (float): Fraction of training data used for validation

    Returns:
        train_generator, val_generator, test_generator
    """

    # Paths
    train_path = f"{data_dir}/train"
    test_path = f"{data_dir}/test"

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=val_split
    )

    # No augmentation for test data
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Training and validation generators
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    # Test generator
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator