from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50


def create_resnet50(dropout_rate, input_shape=(224, 224, 3)):

    base_model = ResNet50(include_top=False, input_shape=input_shape, weights='imagenet')
    # Freeze the base model
    base_model.trainable = False

    # Build the model
    model = models.Sequential([
        base_model,
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Batch Size 32, learning rate 0.001
def create_resnet50_multi_task(dropout_rate=0.3, input_shape=(224, 224, 3)):
    # Load ResNet50 as the base model for better feature extraction
    base_model = ResNet50(include_top=False, input_shape=input_shape, weights='imagenet')

    base_model.trainable = False

    # Input layer
    inputs = layers.Input(shape=input_shape)

    # Base model
    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)

    #  Dense layers for age features
    age = layers.Dense(256, activation='relu')(x)
    age = layers.Dropout(0.3)(age)
    # Age output
    age_output = layers.Dense(5, activation='softmax', name='age_output')(age)

    # Face output
    face_output = layers.Dense(1, activation='sigmoid', name='face_output')(x)

    # Gender output
    gender_output = layers.Dense(1, activation='sigmoid', name='gender_output')(x)

    # Create the model
    model = models.Model(inputs=inputs, outputs=[face_output, age_output, gender_output])

    return model

def create_resnet50_multi_task_v2(dropout_rate=0.4, input_shape=(224, 224, 3)):

    # Load ResNet50
    base_model = ResNet50(include_top=False, input_shape=input_shape, weights='imagenet')
    base_model.trainable = False

    # Input layer
    inputs = layers.Input(shape=input_shape)

    # Base model output (feature maps)
    x = base_model(inputs, training=False)

    # Face Classification Head (uses pooled features)
    face = layers.GlobalAveragePooling2D()(x)
    face_output = layers.Dense(1, activation='sigmoid', name='face_output')(face)

    # Age branch
    age = x  # Use the feature maps before pooling
    age = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(age)
    age = layers.BatchNormalization()(age)
    age = layers.Dropout(dropout_rate)(age)
    age = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(age)
    age = layers.BatchNormalization()(age)
    age = layers.Dropout(dropout_rate)(age)
    age = layers.GlobalAveragePooling2D()(age)
    age = layers.Dense(256, activation='relu')(age)
    age_output = layers.Dense(5, activation='softmax', name='age_output')(age)

    # Gender branch
    gender = x  # Use the feature maps before pooling
    gender = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(gender)
    gender = layers.BatchNormalization()(gender)
    gender = layers.Dropout(dropout_rate)(gender)
    gender = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(gender)
    gender = layers.BatchNormalization()(gender)
    gender = layers.Dropout(dropout_rate)(gender)
    gender = layers.GlobalAveragePooling2D()(gender)
    gender = layers.Dense(128, activation='relu')(gender)
    gender_output = layers.Dense(1, activation='sigmoid', name='gender_output')(gender)

    model = models.Model(inputs=inputs, outputs=[face_output, age_output, gender_output])

    return model

