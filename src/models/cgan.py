import tensorflow as tf
from tensorflow.keras import layers, models

tf.random.set_seed(42)

def create_generator(latent_dim=100, num_conditions=2):
    # Noise input
    noise_input = layers.Input(shape=(latent_dim,))

    # Conditional input
    label_input = layers.Input(shape=(num_conditions,))

    # Process labels with regularization
    label_embedding = layers.Dense(latent_dim,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01))(label_input)
    label_embedding = layers.LeakyReLU(alpha=0.2)(label_embedding)

    # Concatenate noise and labels
    combined_input = layers.Concatenate()([noise_input, label_embedding])

    # Dense and reshape with gradient clipping
    x = layers.Dense(4 * 4 * 512,
                     kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                     kernel_regularizer=tf.keras.regularizers.l2(0.01))(combined_input)
    x = layers.Reshape((4, 4, 512))(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same',
                               kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same',
                               kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same',
                               kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2DTranspose(32, 4, strides=2, padding='same',
                               kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Final layer
    output = layers.Conv2D(3, 4, padding='same', activation='tanh',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(x)

    return models.Model([noise_input, label_input], output, name='Generator')

def create_discriminator(num_conditions=2):
    # Image input
    img_input = layers.Input(shape=(64, 64, 3))

    # Conditional input
    label_input = layers.Input(shape=(num_conditions,))

    # Process labels
    label_embedding = layers.Dense(64 * 64 * 1,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01))(label_input)
    label_embedding = layers.Reshape((64, 64, 1))(label_embedding)

    # Concatenate image and label channels
    x = layers.Concatenate()([img_input, label_embedding])

    x = layers.Conv2D(64, 4, strides=2, padding='same',
                      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, 4, strides=2, padding='same',
                      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, 4, strides=2, padding='same',
                      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(512, 4, strides=2, padding='same',
                      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    # Classification
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation=None)(x)
    output = layers.Activation('sigmoid')(x)

    return models.Model([img_input, label_input], output, name='Discriminator')

def create_cgan(g_lr=1e-4,d_lr=2e-4,latent_dim=100, num_conditions=2):
    # Create generator and discriminator
    generator = create_generator(latent_dim, num_conditions)
    discriminator = create_discriminator(num_conditions)

    optimizer_g = tf.keras.optimizers.Adam(learning_rate=g_lr, beta_1=0.5, beta_2=0.999)
    optimizer_d = tf.keras.optimizers.Adam(learning_rate=d_lr, beta_1=0.5, beta_2=0.999)

    discriminator.compile(
        optimizer=optimizer_d,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.1),
        metrics=['accuracy']
    )

    # Create the CGAN
    discriminator.trainable = False

    noise_input = layers.Input(shape=(latent_dim,))
    label_input = layers.Input(shape=(num_conditions,))

    generated_images = generator([noise_input, label_input])
    validity = discriminator([generated_images, label_input])

    cgan = models.Model([noise_input, label_input], validity, name='CGAN')

    # Compile CGAN
    cgan.compile(
        optimizer=optimizer_g,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)
    )

    return generator, discriminator, cgan