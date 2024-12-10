import tensorflow as tf
import pandas as pd
import os
import numpy as np

# Parameters
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32
BUFFER_SIZE = 10000
TOTAL_IMAGES = 32000
AUTOTUNE = tf.data.AUTOTUNE

def load_and_filter_attributes(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    image_column = df.columns[0]
    # Filter out images with eyeglasses
    df = df[df['Eyeglasses'] == -1]

    # Create combinations of attributes
    conditions = [
        (df['Male'] == 1) & (df['Smiling'] == 1),   # Male, Smiling
        (df['Male'] == 1) & (df['Smiling'] == -1),  # Male, Not Smiling
        (df['Male'] == -1) & (df['Smiling'] == 1),  # Female, Smiling
        (df['Male'] == -1) & (df['Smiling'] == -1)  # Female, Not Smiling
    ]

    # Calculate target size per group for balanced distribution
    target_per_group = TOTAL_IMAGES // 4

    # Sample equal amounts from each group
    filtered_dfs = []
    for i, condition in enumerate(conditions):
        group_size = sum(condition)
        print(f"Group {i} original size: {group_size}")
        sample_size = min(target_per_group, group_size)
        group_df = df[condition].sample(n=sample_size)
        filtered_dfs.append(group_df)
        print(f"Group {i} sampled size: {len(group_df)}")

    # Combine all filtered groups
    final_df = pd.concat(filtered_dfs)
    print(f"Total images in final dataset: {len(final_df)}")

    # Create labels array
    labels = np.stack([
        final_df['Male'].values,
        final_df['Smiling'].values
    ], axis=1)

    # Convert -1 to 0 for binary labels
    labels = (labels + 1) // 2

    # Get image filenames from the first column
    image_filenames = final_df[image_column].values

    return image_filenames, labels

def preprocess_image(image_path, label):
    try:
        # Read and decode image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)

        # Center crop
        height = tf.shape(image)[0]
        width = tf.shape(image)[1]
        crop_size = tf.minimum(height, width)

        # Calculate crop offsets
        offset_height = (height - crop_size) // 2
        offset_width = (width - crop_size) // 2

        # Perform the crop
        image = tf.image.crop_to_bounding_box(
            image,
            offset_height,
            offset_width,
            crop_size,
            crop_size
        )
        image = tf.image.resize(
            image,
            [IMG_HEIGHT, IMG_WIDTH],
            method=tf.image.ResizeMethod.BILINEAR,
            preserve_aspect_ratio=False,
            antialias=True
        )

        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)

        # Normalize to [-1, 1]
        image = (image / 127.5) - 1.0

        # Ensure shape is correct
        image = tf.ensure_shape(image, [IMG_HEIGHT, IMG_WIDTH, 3])

        return image, label

    except tf.errors.InvalidArgumentError as e:
        tf.print(f"Error processing image: {image_path}")
        return tf.zeros([IMG_HEIGHT, IMG_WIDTH, 3], dtype=tf.float32), label

def create_cgan_dataset(data_dir, csv_path, batch_size=BATCH_SIZE):
    if not os.path.exists(data_dir) or not os.path.exists(csv_path):
        raise ValueError(f"Data or CSV directory does not exist")

    # Get filtered image filenames and labels
    image_filenames, labels = load_and_filter_attributes(csv_path)

    # Create full image paths
    image_paths = [os.path.join(data_dir, filename) for filename in image_filenames]

    # Verify few paths
    for path in image_paths[:5]:
        if not os.path.exists(path):
            print(f"Warning: Image not found at {path}")
        else:
            print(f"Found image: {path}")

    # Create tensorflow datasets
    path_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    label_dataset = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((path_dataset, label_dataset))

    # Apply preprocessing
    dataset = dataset.map(
        preprocess_image,
        num_parallel_calls=AUTOTUNE
    )

    # Filter out failed images
    dataset = dataset.filter(
        lambda x, y: tf.reduce_sum(tf.abs(x)) > 0
    )

    # Performance optimizations
    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset

# Usage
DATA_DIR = '../../data/img_align_celeba/img_align_celeba'
CSV_PATH = '../../data/img_align_celeba/list_attr_celeba.csv'
dataset = create_cgan_dataset(DATA_DIR, CSV_PATH)