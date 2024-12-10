import tensorflow as tf
import pandas as pd
import os
import numpy as np
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness, GaussianNoise


# Mapping for age groups
age_group_map = {
    '18-29': 0,
    '30-39': 1,
    '40-49': 2,
    '50-59': 3,
    'more than 60': 4
}

# Model specific preprocessing configurations
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

# Adjust based on chosen model
selected_model_number = 4


# Data Augmentation Function

def get_augmentation():
    return tf.keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.1),
        RandomContrast(0.1),
        RandomBrightness(0.2),
        GaussianNoise(0.05)
    ], name='data_augmentation')


# Function to Load and Clean CSV Data

def load_and_clean_csv(csv_path):
    df = pd.read_csv(csv_path)

    print(df.head())

    # Remove rows with missing labels
    initial_count = len(df)
    df = df.dropna(subset=['face_label', 'age_label', 'gender_label'])
    print(f"Dropped {initial_count - len(df)} rows due to missing labels.")

    # Normalize and standardize image paths
    df['image_path'] = df['image_path'].apply(lambda x: os.path.normpath(x))

    # Keep only rows where image file exists
    existing_images = df['image_path'].apply(lambda x: os.path.exists(x))
    valid_image_count = existing_images.sum()
    df = df[existing_images]
    print(f"After removing non-existent images, {len(df)} rows remain.")

    # Convert face_label and gender_label to integers
    df['face_label'] = df['face_label'].astype(float).astype(int)
    df['gender_label'] = df['gender_label'].astype(float).astype(int)

    # Map exact age to age groups only if face_label == 1
    def map_age_to_group(row):
        if row['face_label'] == 1:
            try:
                age = int(row['age_label'])
            except:
                return -1
            if 18 <= age <= 29:
                return age_group_map['18-29']
            elif 30 <= age <= 39:
                return age_group_map['30-39']
            elif 40 <= age <= 49:
                return age_group_map['40-49']
            elif 50 <= age <= 59:
                return age_group_map['50-59']
            elif age >= 60:
                return age_group_map['more than 60']
            else:
                return -1  # For ages below 18 or invalid
        else:
            return -1  # For non face images

    df['age_group'] = df.apply(map_age_to_group, axis=1).astype(int)

    # Remove rows with invalid age groups only if face_label == 1
    before_age_filter = len(df)
    df_valid_age = df[(df['face_label'] == 0) | (df['age_group'] != -1)]
    removed_age = before_age_filter - len(df_valid_age)
    print(f"Removed {removed_age} rows due to invalid age groups.")
    df = df_valid_age

    print(f"Final DataFrame for {os.path.basename(csv_path)} after cleaning:")
    print(df.head())

    return df


# Function to Encode Labels

def encode_labels(df):
    # One-hot encode age groups
    num_age_classes = len(age_group_map)
    age_one_hot = np.eye(num_age_classes)[df['age_group'].values]
    # Convert to list of lists
    df['age_one_hot'] = age_one_hot.tolist()

    # Binary encode gender labels handle non face images
    # For non face images set gender_binary to 0.0
    df['gender_binary'] = df.apply(lambda row: float(row['gender_label']) if row['face_label'] == 1 else 0.0, axis=1)

    print("DataFrame after encoding labels:")
    print(df[['age_group', 'age_one_hot', 'gender_binary']].head())

    return df


# Function to Preprocess Images

def preprocess_image(image_path, face_label, age_one_hot, gender_binary, preprocess_func, target_size, augmentation):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = preprocess_func(image)
    if augmentation:
        image = augmentation(image)

    # Convert labels to tensors
    face_label = tf.cast(face_label, tf.float32)
    age_one_hot = tf.convert_to_tensor(age_one_hot, dtype=tf.float32)
    gender_binary = tf.cast(gender_binary, tf.float32)

    # Sample weights to only compute age and gender for face images
    is_face = tf.cast(tf.equal(face_label, 1.0), tf.float32)
    sample_weights = {
        'age_output': is_face,
        'gender_output': is_face
    }

    labels = {
        'face_output': face_label,
        'age_output': age_one_hot,
        'gender_output': gender_binary
    }

    return image, labels, sample_weights


# Function to Create TensorFlow Dataset

def create_tf_dataset(df, preprocess_func, target_size, augmentation, batch_size, shuffle=False):
    if df.empty:
        print("Warning: DataFrame is empty. Returning an empty dataset.")
        return tf.data.Dataset.from_tensor_slices(([], [], [], []))

    image_paths = df['image_path'].values
    face_labels = df['face_label'].values
    age_one_hots = df['age_one_hot'].values
    gender_binaries = df['gender_binary'].values

    age_one_hots = [list(aoh) for aoh in age_one_hots]

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, face_labels, age_one_hots, gender_binaries))

    if shuffle:
        buffer_size = max(len(df), 1)
        dataset = dataset.shuffle(buffer_size=buffer_size)

    def map_fn(image_path, face_label, age_one_hot, gender_binary):
        return preprocess_image(image_path, face_label, age_one_hot, gender_binary,
                                preprocess_func, target_size, augmentation)

    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# Function to Prepare Datasets

def prepare_datasets(output_base_dir, model_map, selected_model_number, batch_size=32):
    print(model_map[selected_model_number]['name'])
    preprocess_func = model_map[selected_model_number]['preprocess']
    target_size = model_map[selected_model_number]['input_size']
    augmentation = get_augmentation()

    splits = ['train', 'validation', 'test']
    datasets = {}

    for split in splits:
        csv_path = os.path.join(output_base_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            print(f"\nCSV file for {split} does not exist at {csv_path}. Skipping...")
            datasets[split] = tf.data.Dataset.from_tensor_slices(([], [], [], []))
            continue

        print(f"\nLoading and cleaning {split} data from {csv_path}...")
        df = load_and_clean_csv(csv_path)
        print(f"Encoding labels for {split} data...")
        df = encode_labels(df)

        if df.empty:
            print(f"No valid data found for {split} after cleaning. Skipping dataset creation.")
            datasets[split] = tf.data.Dataset.from_tensor_slices(([], [], [], []))
            continue

        print(f"Creating TensorFlow dataset for {split}...")
        if split == 'train':
            ds = create_tf_dataset(df, preprocess_func, target_size, augmentation, batch_size, shuffle=True)
        else:
            ds = create_tf_dataset(df, preprocess_func, target_size, None, batch_size, shuffle=False)

        datasets[split] = ds
        print(f"{split.capitalize()} dataset prepared with {len(df)} samples.\n")

    return datasets.get('train'), datasets.get('validation'), datasets.get('test')


output_base_dir = '../../data/MultiTaskBalanced'


batch_size = 32
# Prepare the datasets
train_dataset, val_dataset, test_dataset = prepare_datasets(
    output_base_dir=output_base_dir,
    model_map=MODEL_MAP,
    selected_model_number=selected_model_number,
    batch_size=batch_size
)


