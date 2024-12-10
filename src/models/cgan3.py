import tensorflow as tf
from tensorflow.keras import layers, models

def create_generator(latent_dim=100, num_conditions=2):
    noise_input = layers.Input(shape=(latent_dim,))
    label_input = layers.Input(shape=(num_conditions,))

    # Label embedding
    label_embedding = layers.Dense(latent_dim)(label_input)
    label_embedding = layers.LeakyReLU(alpha=0.2)(label_embedding)

    # Combine noise and labels
    combined_input = layers.Concatenate()([noise_input, label_embedding])

    # Dense and reshape
    x = layers.Dense(4 * 4 * 512,
                     kernel_initializer='he_normal')(combined_input)
    x = layers.Reshape((4, 4, 512))(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same',
                               kernel_initializer='he_normal')(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same',
                               kernel_initializer='he_normal')(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same',
                               kernel_initializer='he_normal')(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2DTranspose(32, 4, strides=2, padding='same',
                               kernel_initializer='he_normal')(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Output layer
    output = layers.Conv2D(3, 4, padding='same', activation='tanh',
                           kernel_initializer='he_normal')(x)

    return models.Model([noise_input, label_input], output, name='Generator')


def create_discriminator(num_conditions=2):
    img_input = layers.Input(shape=(64, 64, 3))
    label_input = layers.Input(shape=(num_conditions,))

    # Label embedding
    label_embedding = layers.Dense(64 * 64)(label_input)
    label_embedding = layers.Reshape((64, 64, 1))(label_embedding)

    # Concatenate image and label
    x = layers.Concatenate()([img_input, label_embedding])

    x = layers.Conv2D(64, 4, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 4, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(256, 4, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(512, 4, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    output = layers.Dense(1, activation=None, kernel_initializer='he_normal')(x)

    return models.Model([img_input, label_input], output, name='Discriminator')


def create_cgan3(g_lr=1e-4, d_lr=2e-4, latent_dim=100, num_conditions=2):
    generator = create_generator(latent_dim, num_conditions)
    discriminator = create_discriminator(num_conditions)

    optimizer_g = tf.keras.optimizers.Adam(learning_rate=g_lr, beta_1=0.5, beta_2=0.999)
    optimizer_d = tf.keras.optimizers.Adam(learning_rate=d_lr, beta_1=0.5, beta_2=0.999)

    # Compile discriminator
    discriminator.compile(
        optimizer=optimizer_d,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Create CGAN
    discriminator.trainable = False
    noise_input = layers.Input(shape=(latent_dim,))
    label_input = layers.Input(shape=(num_conditions,))
    generated_images = generator([noise_input, label_input])
    validity = discriminator([generated_images, label_input])

    cgan = models.Model([noise_input, label_input], validity, name='CGAN')

    cgan.compile(
        optimizer=optimizer_g,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    )

    return generator, discriminator, cgan
