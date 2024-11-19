from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


def create_mobileNet(dropout_rate, input_shape=(224, 224, 3)):

    base_model = MobileNetV2(include_top=False, input_shape=input_shape, weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_mobileNet_multi_task(dropout_rate=0,input_shape=(224,224,3)):

    base_model = MobileNetV2(include_top=False, input_shape=input_shape, weights='imagenet')
    # Freeze all layers
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    # Face output
    face_output = layers.Dense(1, activation='sigmoid', name='face_output')(x)

    # Age output
    age_output = layers.Dense(5, activation='softmax', name='age_output')(x)

    # gender output
    gender_output = layers.Dense(1, activation='sigmoid', name='gender_output')(x)

    model = models.Model(inputs=inputs, outputs=[face_output, age_output, gender_output])

    return model


def create_mobileNet_multi_task_v2(dropout_rate=0.5, input_shape=(224,224,3)):

    base_model = MobileNetV2(include_top=False, input_shape=input_shape, weights='imagenet')

    # Freeze all layers
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)

    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)

    # Common dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Face output
    face_output = layers.Dense(1, activation='sigmoid', name='face_output')(x)

    # Age branch
    age = layers.Dense(128, activation='relu')(x)
    age = layers.BatchNormalization()(age)
    age = layers.Dropout(dropout_rate)(age)
    age = layers.Dense(64, activation='relu')(age)
    age = layers.BatchNormalization()(age)
    age_output = layers.Dense(5, activation='softmax', name='age_output')(age)

    # Gender branch
    gender = layers.Dense(32, activation='relu')(x)
    gender = layers.BatchNormalization()(gender)
    gender = layers.Dropout(dropout_rate)(gender)
    gender = layers.Dense(16, activation='relu')(gender)
    gender = layers.BatchNormalization()(gender)
    gender_output = layers.Dense(1, activation='sigmoid', name='gender_output')(gender)

    model = models.Model(inputs=inputs, outputs=[face_output, age_output, gender_output])

    return model