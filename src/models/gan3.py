import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, initializers

tf.random.set_seed(42)

def create_generator(latent_dim=120):
    model = tf.keras.Sequential([
        layers.Dense(4 * 4 * 512, use_bias=False, input_shape=(latent_dim,)),
        layers.Reshape((4, 4, 512)),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.3),

        layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.3),

        layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.3),

        layers.Conv2DTranspose(32, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.3),

        # Final layer
        layers.Conv2D(3, 4, padding='same', activation='tanh',
                      kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02))
    ])
    return model

def create_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, 4, strides=2, padding='same',
                      input_shape=(64, 64, 3)),
        layers.LeakyReLU(alpha=0.2),

        layers.GaussianNoise(0.1),

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

def create_gan3(latent_dim=120, learning_rate=2e-4):
    generator = create_generator(latent_dim)
    discriminator = create_discriminator()

    # Separate optimizers
    generator_optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, clipnorm=1.0)
    discriminator_optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, clipnorm=1.0)

    # Compile the discriminator with label smoothing
    discriminator.compile(
        optimizer=discriminator_optimizer,
        loss=losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    discriminator.trainable = False
    gan_input = layers.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = models.Model(gan_input, gan_output, name='GAN')

    # Compile the GAN
    gan.compile(
        optimizer=generator_optimizer,
        loss=losses.BinaryCrossentropy()
    )

    return generator, discriminator, gan
