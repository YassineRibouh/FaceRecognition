import tensorflow as tf
from tensorflow.keras import layers, models

def create_embedding_network(dropout_rate=0.2):
    inputs = layers.Input(shape=(128, 128, 3))
    # First block
    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_uniform')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Second block
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Third block
    x = layers.Conv2D(256, 3, padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, 3, padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(1024, kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Final embedding layer
    x = layers.Dense(300, kernel_initializer='he_uniform')(x)
    # L2 normalize embeddings
    embeddings = layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1),
        output_shape=(300,))(x)
    model = models.Model(inputs, embeddings, name="EmbeddingNetwork")
    return model

def triplet_loss(margin=0.3):
    def loss(y_true, y_pred):
        # Split embeddings
        anchor, positive, negative = tf.split(y_pred, 3, axis=1)
        # Compute distances
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        # Smooth margin
        basic_loss = tf.math.softplus(pos_dist - neg_dist + margin)
        # Add regularization
        regularization = 0.01 * (tf.reduce_mean(tf.square(anchor)) +
                                 tf.reduce_mean(tf.square(positive)) +
                                 tf.reduce_mean(tf.square(negative)))
        return tf.reduce_mean(basic_loss) + regularization
    return loss

def create_triplet_model_v2(dropout_rate):
    input_shape=(128, 128, 3)
    anchor_input = layers.Input(shape=input_shape, name="anchor_input")
    positive_input = layers.Input(shape=input_shape, name="positive_input")
    negative_input = layers.Input(shape=input_shape, name="negative_input")

    encoder = create_embedding_network(dropout_rate)

    # Generate embeddings using shared encoder
    anchor_embedding = encoder(anchor_input)
    positive_embedding = encoder(positive_input)
    negative_embedding = encoder(negative_input)

    # Concatenate embeddings
    outputs = layers.Concatenate(axis=1)([anchor_embedding, positive_embedding, negative_embedding])

    model = models.Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=outputs,
        name="TripletModel"
    )
    return model

def create_and_compile_triplet_v5(dropout_rate=0.2, margin=0.3, learning_rate=1e-5):
    model = create_triplet_model_v2(dropout_rate)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )
    model.compile(
        optimizer=optimizer,
        loss=triplet_loss(margin=margin)
    )
    return model