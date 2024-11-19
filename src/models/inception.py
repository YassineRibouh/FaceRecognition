from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3


def create_inception(dropout_rate, input_shape=(299, 299, 3)):

    base_model = InceptionV3(include_top=False, input_shape=input_shape, weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])
    return model
