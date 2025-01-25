from tensorflow.keras import layers, Model, Input
import tensorflow as tf
from loss_functions import contrastive_loss, contrastive_accuracy

def create_embedding_network(dropout_rate=0.3):

    inputs = Input(shape=(128, 128, 3))

    # First convolutional block
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Second convolutional block
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Third convolutional block
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Fourth convolutional block
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Flatten and dense layers
    x = layers.Flatten()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Final embedding layer with L2 normalization
    x = layers.Dense(128)(x)
    outputs = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)

    # Create the model
    model = Model(inputs, outputs, name="EmbeddingNetwork")

    return model


def create_siamese_network(dropout_rate=0.3):
    embedding_network = create_embedding_network(dropout_rate)

    input_image1 = layers.Input(shape=(128, 128, 3))
    input_image2 = layers.Input(shape=(128, 128, 3))

    embedding1 = embedding_network(input_image1)
    embedding2 = embedding_network(input_image2)

    # Calculate L2 normalized embeddings and distance
    l2_distance = tf.sqrt(tf.reduce_sum(tf.square(embedding1 - embedding2), axis=1, keepdims=True))

    # Create model
    model = Model(
        inputs=[input_image1, input_image2],
        outputs=l2_distance
    )
    return model

def create_and_compile_contrastive_v1(
        dropout_rate=0.3,
        learning_rate=0.001,
):
    # Create model
    model = create_siamese_network(dropout_rate)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=contrastive_loss,
        metrics=[contrastive_accuracy]
    )
    return model