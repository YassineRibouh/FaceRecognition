import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from tensorflow.keras.applications import efficientnet_v2

# Configuration
IDENTITY_FILE = '../../data/img_align_celeba/identity_CelebA.txt'
IMAGE_DIR = '../../data/img_align_celeba/img_align_celeba'
BATCH_SIZE = 32
MAX_IDENTITIES = 1000
SAMPLES_PER_IDENTITY = 6
AUTOTUNE = tf.data.AUTOTUNE

def load_identity_data():
    print("Loading data")
    identity_df = pd.read_csv(IDENTITY_FILE, sep=' ', names=['image_id', 'identity'])
    # Get identities
    identity_counts = identity_df['identity'].value_counts()
    # Filters identities with at least SAMPLES_PER_IDENTITY images
    valid_identities = identity_counts[
        identity_counts >= SAMPLES_PER_IDENTITY
        ].nlargest(MAX_IDENTITIES).index
    filtered_dfs = []
    # randomly sample SAMPLES_PER_IDENTITY images
    for identity in tqdm(valid_identities, desc="Processing identities"):
        samples = identity_df[identity_df['identity'] == identity].sample(
            n=SAMPLES_PER_IDENTITY, random_state=42
        )
        filtered_dfs.append(samples)

    final_df = pd.concat(filtered_dfs, ignore_index=True)
    final_df['image_path'] = IMAGE_DIR + '/' + final_df['image_id']
    # Check if the image exist
    existing_mask = final_df['image_path'].apply(os.path.exists)
    final_df = final_df[existing_mask].reset_index(drop=True)

    print(f"Total images: {len(final_df)}")
    print(f"Number of identities: {len(final_df['identity'].unique())}")

    return final_df

def create_pairs_dataset(identity_df, model_type='custom', is_training=False):
    # Group images
    identity_groups = {
        name: group['image_path'].tolist()
        for name, group in identity_df.groupby('identity')
    }
    identities = list(identity_groups.keys())

    def preprocess_image(image_path):
        # Read image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)

        if model_type == 'efficientnet':
            image = tf.image.resize(image, (128, 128))
            image = efficientnet_v2.preprocess_input(image)
        else:
            image = tf.image.resize(image, (128, 128))
            # [-1, 1]
            image = (image / 127.5) - 1.0

        # Apply augmentations during training
        if is_training:
            image = tf.image.random_brightness(image, 0.2)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            image = tf.image.random_flip_left_right(image)
        return image

    def generate_pairs():
        while True:
            pairs_per_batch = BATCH_SIZE // 2
            anchor_paths = []
            comparison_paths = []
            labels = []
            # Generate positive pairs
            pos_identities = np.random.choice(identities, pairs_per_batch, replace=False)
            for identity in pos_identities:
                imgs = np.random.choice(identity_groups[identity], 2, replace=False)
                anchor_paths.append(imgs[0])
                comparison_paths.append(imgs[1])
                labels.append(1)
            # Generate negative pairs
            for identity in pos_identities:
                diff_identity = np.random.choice(list(set(identities) - {identity}))
                anchor_paths.append(np.random.choice(identity_groups[identity]))
                comparison_paths.append(np.random.choice(identity_groups[diff_identity]))
                labels.append(0)
            yield (anchor_paths, comparison_paths), labels

    # Create dataset
    dataset = tf.data.Dataset.from_generator(
        generate_pairs,
        output_signature=(
            (
                tf.TensorSpec(shape=(BATCH_SIZE,), dtype=tf.string),
                tf.TensorSpec(shape=(BATCH_SIZE,), dtype=tf.string)
            ),
            tf.TensorSpec(shape=(BATCH_SIZE,), dtype=tf.int32)
        )
    )
    # Proces pairs
    def process_pairs(paths, labels):
        anchor_paths, comparison_paths = paths
        # Process images
        anchor_images = tf.map_fn(
            preprocess_image,
            anchor_paths,
            fn_output_signature=tf.float32,
            parallel_iterations=4
        )
        comparison_images = tf.map_fn(
            preprocess_image,
            comparison_paths,
            fn_output_signature=tf.float32,
            parallel_iterations=4
        )
        return (anchor_images, comparison_images), labels
    #final dataset
    dataset = dataset.map(process_pairs, num_parallel_calls=AUTOTUNE)
    if is_training:
        dataset = dataset.shuffle(buffer_size=500)
    dataset = dataset.prefetch(AUTOTUNE)
    dataset = dataset.repeat()

    return dataset
def get_train_val_test_splits(identity_df, model_type='custom', train_ratio=0.7, val_ratio=0.15):
    print(f"Using model type: {model_type}")

    # Split identities
    identities = np.array(list(identity_df['identity'].unique()))
    np.random.shuffle(identities)

    n_train = int(len(identities) * train_ratio)
    n_val = int(len(identities) * val_ratio)

    train_identities = identities[:n_train]
    val_identities = identities[n_train:n_train + n_val]
    test_identities = identities[n_train + n_val:]

    # Create datasets for each split
    train_df = identity_df[identity_df['identity'].isin(train_identities)]
    val_df = identity_df[identity_df['identity'].isin(val_identities)]
    test_df = identity_df[identity_df['identity'].isin(test_identities)]

    train_dataset = create_pairs_dataset(train_df, model_type, is_training=True)
    val_dataset = create_pairs_dataset(val_df, model_type, is_training=False)
    test_dataset = create_pairs_dataset(test_df, model_type, is_training=False)

    return train_dataset, val_dataset, test_dataset

identity_df = load_identity_data()

train_contrastive_dataset, val_contrastive_dataset, test_contrastive_dataset = get_train_val_test_splits(
    identity_df,
    # 'custom' or 'efficientnet'
    model_type='custom')
