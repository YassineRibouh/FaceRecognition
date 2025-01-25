import tensorflow as tf
from tensorflow.keras import layers, models
from loss_functions import triplet_loss, triplet_accuracy
def create_embedding_network(dropout_rate=0.2):
    inputs = layers.Input(shape=(128, 128, 3))
    # First block
    x = layers.Conv2D(32, 3, padding='same', kernel_initializer='he_uniform')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(32, 3, padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Second block
    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Third block
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(dropout_rate)(x)

    # L2 normalize embeddings
    embeddings = layers.Dense(128, kernel_initializer='he_uniform')(x)
    embeddings = layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1),
        output_shape=(128,))(embeddings)

    model = models.Model(inputs, embeddings, name="EmbeddingNetwork")
    return model

def create_triplet_model(dropout_rate):
    input_shape=(128, 128, 3)
    anchor_input = layers.Input(shape=input_shape, name="anchor_input")
    positive_input = layers.Input(shape=input_shape, name="positive_input")
    negative_input = layers.Input(shape=input_shape, name="negative_input")

    embedding_network = create_embedding_network(dropout_rate)
    # Share weights
    anchor_embedding = embedding_network(anchor_input)
    positive_embedding = embedding_network(positive_input)
    negative_embedding = embedding_network(negative_input)

    outputs = layers.Concatenate(axis=1)([anchor_embedding, positive_embedding, negative_embedding])
    model = models.Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=outputs,
        name="TripletModel"
    )
    return model

def create_and_compile_triplet_v4(dropout_rate=0.2, margin=0.3,learning_rate=1e-5):
    model = create_triplet_model(dropout_rate)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )
    model.compile(
        optimizer=optimizer,
        loss=triplet_loss(margin=margin),
        metrics=[triplet_accuracy]
    )
    return model