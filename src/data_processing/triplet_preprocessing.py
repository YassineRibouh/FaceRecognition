import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.keras.applications import efficientnet_v2

# Configuration
IDENTITY_FILE = '../../data/img_align_celeba/identity_CelebA.txt'
IMAGE_DIR = '../../data/img_align_celeba/img_align_celeba'
BATCH_SIZE = 32
MAX_IDENTITIES = 1000
MIN_SAMPLES_PER_IDENTITY = 6
AUTOTUNE = tf.data.AUTOTUNE

def load_identity_data():
    print("Loading data")
    identity_df = pd.read_csv(IDENTITY_FILE, sep=' ', names=['image_id', 'identity'])
    identity_df['image_path'] = IMAGE_DIR + '/' + identity_df['image_id']
    # Get identities
    identity_counts = identity_df['identity'].value_counts()
    valid_identities = identity_counts[
        identity_counts >= MIN_SAMPLES_PER_IDENTITY
        ].nlargest(MAX_IDENTITIES).index
    filtered_df = identity_df[identity_df['identity'].isin(valid_identities)]

    existing_mask = filtered_df['image_path'].apply(os.path.exists)
    final_df = filtered_df[existing_mask].reset_index(drop=True)

    print(f"Total images: {len(final_df)}")
    print(f"Number of identities: {len(final_df['identity'].unique())}")

    return final_df

def create_triplets_dataset(identity_df, model_type='custom', is_training=False):
    # Group images by identity
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

    def generate_triplets():
        while True:
            anchor_paths = []
            positive_paths = []
            negative_paths = []

            # Generate triplets
            for _ in range(BATCH_SIZE):
                # Select anchor identity
                anchor_identity = np.random.choice(identities)
                anchor_images = identity_groups[anchor_identity]

                # Select anchor and positive
                anchor_img, positive_img = np.random.choice(anchor_images, 2, replace=False)

                # Select negative from different identity
                negative_identity = np.random.choice(list(set(identities) - {anchor_identity}))
                negative_img = np.random.choice(identity_groups[negative_identity])

                anchor_paths.append(anchor_img)
                positive_paths.append(positive_img)
                negative_paths.append(negative_img)

            yield anchor_paths, positive_paths, negative_paths

    # Create dataset
    dataset = tf.data.Dataset.from_generator(
        generate_triplets,
        output_signature=(
            tf.TensorSpec(shape=(BATCH_SIZE,), dtype=tf.string),
            tf.TensorSpec(shape=(BATCH_SIZE,), dtype=tf.string),
            tf.TensorSpec(shape=(BATCH_SIZE,), dtype=tf.string)
        )
    )

    # Process triplets
    def process_triplets(anchor_paths, positive_paths, negative_paths):
        # Process images
        anchor_images = tf.map_fn(
            preprocess_image,
            anchor_paths,
            fn_output_signature=tf.float32,
            parallel_iterations=4
        )
        positive_images = tf.map_fn(
            preprocess_image,
            positive_paths,
            fn_output_signature=tf.float32,
            parallel_iterations=4
        )
        negative_images = tf.map_fn(
            preprocess_image,
            negative_paths,
            fn_output_signature=tf.float32,
            parallel_iterations=4
        )

        return anchor_images, positive_images, negative_images

    # final dataset
    dataset = dataset.map(process_triplets, num_parallel_calls=AUTOTUNE)
    if is_training:
        dataset = dataset.shuffle(buffer_size=500)

    # Prepare dataset for model input
    def prepare_batch(anchor, positive, negative):
        return (
            {
                'anchor_input': anchor,
                'positive_input': positive,
                'negative_input': negative
            },
            # Added dummy labels because the models expects a dictionary with named inputs
            tf.zeros((tf.shape(anchor)[0], 1))
        )
    dataset = dataset.map(prepare_batch, num_parallel_calls=AUTOTUNE)
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

    train_dataset = create_triplets_dataset(train_df, model_type, is_training=True)
    val_dataset = create_triplets_dataset(val_df, model_type, is_training=False)
    test_dataset = create_triplets_dataset(test_df, model_type, is_training=False)

    return train_dataset, val_dataset, test_dataset

identity_df = load_identity_data()

train_triplet_dataset, val_triplet_dataset, test_triplet_dataset = get_train_val_test_splits(
    identity_df,
    # 'custom' or 'efficientnet'
    model_type='custom')