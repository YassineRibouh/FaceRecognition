from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2B0, EfficientNetV2B3

# Face Classification

# -------------------------------------------------------------------------------------------------------

# Batch size 32, learning rate 0.001, dopout 0.3
def create_efficientnetb0(dropout_rate, input_shape=(224, 224, 3)):
    base_model = EfficientNetV2B0(include_top=False, input_shape=input_shape, weights='imagenet')
    base_model.trainable = False  # Freeze the base model

    # Build the model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Batch size 32, learning rate 0.001, dopout 0.3
def create_efficientnetb3(dropout_rate, input_shape=(224, 224, 3)):
    base_model = EfficientNetV2B3(include_top=False, input_shape=input_shape, weights='imagenet')
    base_model.trainable = False  # Freeze the base model

    # Build the model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


# -------------------------------------------------------------------------------------------------------
# Multi Task

# EfficientNet B0

# Batch size 32, learning rate 0.001, dropout 0.3
def create_efficientnetb0_multi_task(dropout_rate, input_shape=(224, 224, 3)):


    base_model = EfficientNetV2B0(include_top=False, input_shape=input_shape, weights='imagenet')
    # Freeze all layers
    base_model.trainable = False

    # Input layer
    inputs = layers.Input(shape=input_shape)

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Face output
    face_output = layers.Dense(1, activation='sigmoid', name='face_output')(x)

    # Age output
    age_output = layers.Dense(5, activation='softmax', name='age_output')(x)

    # Gender output
    gender_output = layers.Dense(1, activation='sigmoid', name='gender_output')(x)

    model = models.Model(inputs=inputs, outputs=[face_output, age_output, gender_output])

    return model


# Batch size 32, learning rate 0.0001, dopout 0.3
def create_efficientnetb0_multi_task_v2(dropout_rate, input_shape=(224, 224, 3)):


    base_model = EfficientNetV2B0(include_top=False, input_shape=input_shape, weights='imagenet')

    # Unfreeze the 100 top layers
    base_model.trainable = True
    for layer in base_model.layers[:-100]:
        layer.trainable = False

    inputs = layers.Input(shape=input_shape)

    # Base model output
    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Age branch
    age = layers.Dense(512, activation='relu')(x)
    age = layers.BatchNormalization()(age)
    age = layers.Dropout(dropout_rate)(age)
    age = layers.Dense(256, activation='relu')(age)
    age = layers.BatchNormalization()(age)
    age_output = layers.Dense(5, activation='softmax', name='age_output')(age)

    # Face branch
    face_output = layers.Dense(1, activation='sigmoid', name='face_output')(x)

    # Gender branch
    gender = layers.Dense(256, activation='relu')(x)
    gender = layers.BatchNormalization()(gender)
    gender = layers.Dropout(dropout_rate)(gender)
    gender = layers.Dense(128, activation='relu')(gender)
    gender = layers.BatchNormalization()(gender)
    gender_output = layers.Dense(1, activation='sigmoid', name='gender_output')(gender)

    model = models.Model(inputs=inputs, outputs=[face_output, age_output, gender_output])

    return model

# EfficientNet B3

# Batch size 32, learning rate 0.001, dropout 0.3
def create_efficientnetb3_multi_task(dropout_rate, input_shape=(224, 224, 3)):
    # Load EfficientNetV2B0 as the base model
    base_model = EfficientNetV2B3(include_top=False, input_shape=input_shape, weights='imagenet')
    # Freeze all layers
    base_model.trainable = False

    # Input layer
    inputs = layers.Input(shape=input_shape)

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Face output
    face_output = layers.Dense(1, activation='sigmoid', name='face_output')(x)

    # Age output
    age_output = layers.Dense(5, activation='softmax', name='age_output')(x)

    # Gender output
    gender_output = layers.Dense(1, activation='sigmoid', name='gender_output')(x)

    model = models.Model(inputs=inputs, outputs=[face_output, age_output, gender_output])

    return model


# Batchsize 64, learning rate 0.0001, dropout 0.4
def create_efficientnetb3_multi_task_v2(dropout_rate=0.4, input_shape=(224, 224, 3)):
    # Load EfficientNetV2B3 as the base model
    base_model = EfficientNetV2B3(include_top=False, input_shape=input_shape, weights='imagenet')

    # Freeze all layers
    base_model.trainable = False

    # Unfreeze the top 10 layers
    for layer in base_model.layers[-10:]:
        layer.trainable = True

    inputs = layers.Input(shape=input_shape)

    # Enable training mode for unfreezing
    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Face output
    face_output = layers.Dense(1, activation='sigmoid', name='face_output')(x)

    # Age output
    age_output = layers.Dense(5, activation='softmax', name='age_output')(x)

    # gender Output
    gender_output = layers.Dense(1, activation='sigmoid', name='gender_output')(x)

    model = models.Model(inputs=inputs, outputs=[face_output, age_output, gender_output])

    return model
