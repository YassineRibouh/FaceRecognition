import tensorflow as tf

def contrastive_loss(y_true, y_pred, margin=1.0):
    """
    Calculates the contrastive loss between pairs of embeddings.

    Args:
        y_true: Binary labels (1 for similar pairs, 0 for dissimilar pairs)
        y_pred: Predicted distances between pairs
        margin: Margin for negative pairs (default: 1.0)

    Returns:
        Mean contrastive loss value
    """
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
    """
    Calculates binary accuracy for contrastive learning based on a distance threshold.

    Args:
        y_true: Binary labels (1 for similar pairs, 0 for dissimilar pairs)
        y_pred: Predicted distances between pairs
        threshold: Distance threshold for binary prediction (default: 0.5)

    Returns:
        Binary accuracy value
    """
    # Convert predictions to binary predictions using threshold
    predictions = tf.less(y_pred, threshold)
    # Convert to same type for comparison
    y_true = tf.cast(y_true, tf.bool)
    # Calculate accuracy
    matches = tf.equal(predictions, y_true)
    return tf.reduce_mean(tf.cast(matches, tf.float32))

def triplet_accuracy(y_true, y_pred, margin=0.3):
    """
    Calculates accuracy for triplet learning by checking if anchor-positive distance
    is less than anchor-negative distance by the margin.

    Args:
        y_true: Dummy labels (not used but required by Keras)
        y_pred: Concatenated embeddings [anchor, positive, negative]
        margin: Minimum required difference between positive and negative distances

    Returns:
        Binary accuracy value indicating % of valid triplets
    """
    # Split embeddings
    anchor, positive, negative = tf.split(y_pred, 3, axis=1)

    # Compute distances
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

    # Check if pos_dist + margin < neg_dist
    valid_triplets = tf.less(pos_dist + margin, neg_dist)
    return tf.reduce_mean(tf.cast(valid_triplets, tf.float32))

def triplet_loss(margin=0.3):
    """
    Creates a triplet loss function with the specified margin.

    Args:
        margin: Margin value for triplet loss (default: 0.3)

    Returns:
        Triplet loss function that can be used in model compilation
    """
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