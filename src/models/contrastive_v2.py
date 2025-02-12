import tensorflow as tf
from tensorflow.keras import layers, models
from loss_functions import contrastive_loss, contrastive_accuracy

def create_residual_block(x, filters, dropout_rate=0.3):
    shortcut = layers.Conv2D(filters, (1, 1), padding='same')(x)

    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Add()([x, shortcut])
    return layers.Activation('relu')(x)

def create_embedding_network_v2(dropout_rate=0.3):
    inputs = layers.Input(shape=(128, 128, 3))

    # Initial convolution with more filters
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # Residual blocks with increasing filters
    x = create_residual_block(x, 64, dropout_rate)
    x = layers.MaxPooling2D((2, 2))(x)

    x = create_residual_block(x, 128, dropout_rate)
    x = layers.MaxPooling2D((2, 2))(x)

    x = create_residual_block(x, 256, dropout_rate)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers with skip connection
    dense1 = layers.Dense(512)(x)
    dense1 = layers.BatchNormalization()(dense1)
    dense1 = layers.Activation('relu')(dense1)
    dense1 = layers.Dropout(dropout_rate)(dense1)

    dense2 = layers.Dense(256)(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    dense2 = layers.Activation('relu')(dense2)
    dense2 = layers.Dropout(dropout_rate)(dense2)

    # Dense skip connection
    dense_skip = layers.Dense(256)(x)
    x = layers.Add()([dense2, dense_skip])

    # Final embedding layer
    x = layers.Dense(128)(x)
    outputs = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)

    return models.Model(inputs, outputs)

def create_siamese_network_v2(dropout_rate=0.3):
    embedding_network = create_embedding_network_v2(dropout_rate)

    input_image1 = layers.Input(shape=(128, 128, 3))
    input_image2 = layers.Input(shape=(128, 128, 3))

    embedding1 = embedding_network(input_image1)
    embedding2 = embedding_network(input_image2)

    # Calculate L2 distance
    l2_distance = tf.sqrt(tf.reduce_sum(tf.square(embedding1 - embedding2), axis=1, keepdims=True))
    return models.Model(
        inputs=[input_image1, input_image2],
        outputs=l2_distance
    )

def create_and_compile_contrastive_v2(
        dropout_rate=0.3,
        learning_rate=0.0005
):
    model = create_siamese_network_v2(dropout_rate)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )
    model.compile(
        optimizer=optimizer,
        loss=contrastive_loss,
        metrics=[contrastive_accuracy]
    )

    return model