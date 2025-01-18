import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2B3
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout

def create_embedding_network(dropout_rate=0.3, num_layers_to_unfreeze=25):
    # Load EfficientNetB3
    base_model = EfficientNetV2B3(
        weights='imagenet',
        input_shape=(128, 128, 3),
        include_top=False,
        pooling='avg'
    )
    # Freeze the entire model
    base_model.trainable = False

    # Unfreeze the layers
    for layer in base_model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True


    inputs = layers.Input(shape=(128, 128, 3))
    x = base_model(inputs, training=False)

    x = Flatten()(x)

    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Final dense
    embeddings = Dense(128)(x)
    embeddings = layers.Lambda(
        lambda z: tf.math.l2_normalize(z, axis=1),
        output_shape=(128,)
    )(embeddings)
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

def create_and_compile_triplet_v6(dropout_rate=0.3, margin=0.3, learning_rate=1e-4):
    model = create_triplet_model(dropout_rate)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )
    model.compile(
        optimizer=optimizer,
        loss=triplet_loss(margin=margin)
    )
    return model