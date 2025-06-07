import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def create_mobilenetv2_model(input_shape=(224, 224, 3), num_classes=3, fine_tune_at=None):
    """
    Create a MobileNetV2-based model for transfer learning.

    Args:
        input_shape (tuple): Shape of input images.
        num_classes (int): Number of classes to classify.
        fine_tune_at (int or None): If specified, unfreezes the model from this layer index onward.

    Returns:
        tf.keras.Model: Compiled model ready for training.
    """
    base_model = MobileNetV2(input_shape=input_shape,
                             include_top=False,
                             weights="imagenet")

    base_model.trainable = False  # Freeze entire base model

    # Unfreeze later layers if fine-tuning
    if fine_tune_at is not None:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model