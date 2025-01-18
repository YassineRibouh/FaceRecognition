import tensorflow as tf
from prompt_toolkit.input import Input
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.applications import efficientnet_v2


# Trained using 224x224 input shape
def create_embedding_network(dropout_rate=0.3):
    # Load pretrained model
    base_model = efficientnet_v2.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    base_model.trainable = False
    inputs = Input(shape=(224, 224, 3))

    # Base model
    x = base_model(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # First dense block
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    # Second dense block
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    # Final embedding layer with L2 normalization
    x = layers.Dense(128)(x)
    outputs = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)

    # Create the model
    model = Model(inputs, outputs, name="EmbeddingNetwork")

    return model

def create_siamese_network(dropout_rate=0.3):
    embedding_network = create_embedding_network(dropout_rate)

    # Create inputs
    input_image1 = layers.Input(shape=(224, 224, 3))
    input_image2 = layers.Input(shape=(224, 224, 3))

    # Get embeddings for both images
    embedding1 = embedding_network(input_image1)
    embedding2 = embedding_network(input_image2)

    # Calculate L2 distance between embeddings
    l2_distance = tf.sqrt(tf.reduce_sum(tf.square(embedding1 - embedding2), axis=1, keepdims=True))

    # Create model
    model = models.Model(
        inputs=[input_image1, input_image2],
        outputs=l2_distance
    )

    return model

def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, tf.float32)
    # Square the predictions (distances)
    squared_pred = tf.square(y_pred)
    # Loss for similar pairs
    positive_loss = y_true * squared_pred
    # Loss for dissimilar pairs with margin
    negative_loss = (1.0 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
    # Return mean loss
    return tf.reduce_mean(positive_loss + negative_loss) / 2.0

def contrastive_accuracy(y_true, y_pred, threshold=0.5):
    # Convert predictions to binary predictions using threshold
    predictions = tf.less(y_pred, threshold)
    # Convert to same type for comparison
    y_true = tf.cast(y_true, tf.bool)
    # Calculate accuracy
    matches = tf.equal(predictions, y_true)
    return tf.reduce_mean(tf.cast(matches, tf.float32))

def create_and_compile_contrastive_v3(
        dropout_rate=0.3,
        learning_rate=0.0001,
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