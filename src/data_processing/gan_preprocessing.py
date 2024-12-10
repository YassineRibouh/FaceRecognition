import tensorflow as tf
import os

# Parameters
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32
BUFFER_SIZE = 10000
MAX_IMAGES = 40000
AUTOTUNE = tf.data.AUTOTUNE

def preprocess_image(image_path):
    try:
        # Read and decode image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)

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

        return image

    except tf.errors.InvalidArgumentError as e:
        tf.print(f"Error processing image: {image_path}")
        return tf.zeros([IMG_HEIGHT, IMG_WIDTH, 3], dtype=tf.float32)

def create_dataset(data_dir, batch_size=BATCH_SIZE):
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist")

    # List all image files
    pattern = os.path.join(data_dir, "*")
    filenames = tf.data.Dataset.list_files(pattern, shuffle=True)

    filenames = filenames.take(MAX_IMAGES)

    # Create dataset
    dataset = filenames.map(
        preprocess_image,
        num_parallel_calls=AUTOTUNE
    )

    # Filter out failed images
    dataset = dataset.filter(
        lambda x: tf.reduce_sum(tf.abs(x)) > 0
    )

    # Performance optimizations
    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset

DATA_DIR = '../../data/img_align_celeba/img_align_celeba'
CSV_DIR = '../../data/list_attr_celeba/list_attr_celeba.csv'
dataset = create_dataset(DATA_DIR)
