import tensorflow as tf

# Model-specific preprocessing configurations
MODEL_MAP = {
    1: {
        'name': 'efficientnet_v2',
        'preprocess': tf.keras.applications.efficientnet_v2.preprocess_input,
        'input_size': (224, 224)
    },
    2: {
        'name': 'resnet50',
        'preprocess': tf.keras.applications.resnet50.preprocess_input,
        'input_size': (224, 224)
    },
    3: {
        'name': 'inception_v3',
        'preprocess': tf.keras.applications.inception_v3.preprocess_input,
        'input_size': (299, 299)
    },
    4: {
        'name': 'mobilenet_v2',
        'preprocess': tf.keras.applications.mobilenet_v2.preprocess_input,
        'input_size': (224, 224)
    }
}
selected_model_number = 1

train_dir = '../../data/augmented_data/face_classification/train/'
test_dir = '../../data/augmented_data/face_classification/test/'
validation_dir = '../../data/augmented_data/face_classification/val/'

# Parameters
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE


# Data Augmentation

def get_face_augmentation():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(factor=0.1),
        tf.keras.layers.RandomContrast(factor=0.1),
        tf.keras.layers.GaussianNoise(0.05)
    ])
    return data_augmentation


# Dynamic Parsing Functions


# Parsing Functions
def parse_image(image, label, preprocess_func, target_size):
    image = tf.image.resize(image, target_size)
    image = preprocess_func(image)
    return image, label


def parse_and_augment(image, label, preprocess_func, data_augmentation, target_size):
    image = tf.image.resize(image, target_size)
    image = data_augmentation(image, training=True)
    image = preprocess_func(image)
    return image, label


def load_dataset(directory, model_number, augment=False, shuffle=True, batch_size=batch_size, autotune=AUTOTUNE):
    if model_number not in MODEL_MAP:
        raise ValueError(f"Model number {model_number} is not defined in MODEL_MAP.")

    # Get model specific configurations
    preprocess_func = MODEL_MAP[model_number]['preprocess']
    target_size = MODEL_MAP[model_number]['input_size']

    # Initialize data augmentation
    data_augmentation = get_face_augmentation() if augment else None

    # Load dataset from directory
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='binary',
        batch_size=batch_size,
        image_size=target_size,
        shuffle=shuffle
    )

    # Apply augmentation and preprocessing
    if augment and data_augmentation:
        dataset = dataset.map(
            lambda x, y: parse_and_augment(x, y, preprocess_func, data_augmentation, target_size),
            num_parallel_calls=autotune
        )
    else:
        dataset = dataset.map(
            lambda x, y: parse_image(x, y, preprocess_func, target_size),
            num_parallel_calls=autotune
        )

    return dataset.prefetch(buffer_size=autotune)


# Load Datasets with Selected Model Number
train_data = load_dataset(train_dir, model_number=selected_model_number, augment=True, shuffle=True)
validation_data = load_dataset(validation_dir, model_number=selected_model_number, augment=False, shuffle=False)
test_data = load_dataset(test_dir, model_number=selected_model_number, augment=False, shuffle=False)
