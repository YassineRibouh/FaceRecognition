import tensorflow as tf
from tensorflow.keras import layers, models

tf.random.set_seed(42)

def create_generator(latent_dim=100):
    model = tf.keras.Sequential([
        layers.Dense(4 * 4 * 512, use_bias=False, input_shape=(latent_dim,)),
        layers.Reshape((4, 4, 512)),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(32, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        # Final layer
        layers.Conv2D(3, 4, padding='same', activation='tanh',
                      kernel_initializer='glorot_normal')
    ])
    return model

def create_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, 4, strides=2, padding='same',
                      input_shape=(64, 64, 3)),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2D(128, 4, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.4),

        layers.Conv2D(256, 4, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.4),

        layers.Conv2D(512, 4, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.4),

        # Classification
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def create_gan2(latent_dim=100,learning_rate=2e-3):
    generator = create_generator(latent_dim)
    discriminator = create_discriminator()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, clipvalue=1.0)

    # Compile the discriminator
    discriminator.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    discriminator.trainable = False
    gan_input = layers.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = models.Model(gan_input, gan_output, name='GAN')

    # Compile the GAN
    gan.compile(
        optimizer=optimizer,
        loss='binary_crossentropy'
    )

    return generator, discriminator, gan
