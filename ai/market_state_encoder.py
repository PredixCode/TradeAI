import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
import os

class Sampling(layers.Layer):
    """
    Custom Keras layer to perform the reparameterization trick.
    It uses z_mean and z_log_var to sample z, the latent vector.
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class MarketStateEncoderVAE(Model):
    """
    A Variational Autoencoder (VAE) designed to learn a compressed latent
    representation of the market state from various financial indicators.
    """
    def __init__(self, input_dim: int, latent_dim: int = 16, **kwargs):
        super(MarketStateEncoderVAE, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        
        print(f"MarketStateEncoderVAE initialized. Input Dim: {input_dim}, Latent Dim: {latent_dim}")

    def _build_encoder(self) -> Model:
        """Builds the encoder model that maps inputs to the latent space."""
        encoder_inputs = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(128, activation="relu")(encoder_inputs)
        x = layers.Dense(64, activation="relu")(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        
        encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        print("Encoder model built successfully.")
        encoder.summary()
        return encoder

    def _build_decoder(self) -> Model:
        """Builds the decoder model that maps latent space back to original inputs."""
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(64, activation="relu")(latent_inputs)
        x = layers.Dense(128, activation="relu")(x)
        decoder_outputs = layers.Dense(self.input_dim, activation="sigmoid")(x) # Sigmoid for scaled data [0,1]
        
        decoder = Model(latent_inputs, decoder_outputs, name="decoder")
        print("Decoder model built successfully.")
        decoder.summary()
        return decoder

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        """Defines the custom logic for one training step."""
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Calculate reconstruction loss (how well we can recreate the input)
            # --- THE FIX IS HERE ---
            # We replace the outdated 'tf.keras.losses.mean_squared_error' with its
            # direct mathematical equivalent, which is more robust to version changes.
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(data - reconstruction), axis=-1
                )
            )
            
            # Calculate KL divergence loss (regularizer to keep latent space organized)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            total_loss = reconstruction_loss + kl_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def save(self, filepath: str):
        """Saves the encoder and decoder models."""
        print(f"Saving VAE models to directory: {filepath}")
        os.makedirs(filepath, exist_ok=True)
        self.encoder.save(os.path.join(filepath, 'encoder.keras'))
        self.decoder.save(os.path.join(filepath, 'decoder.keras'))
        print("VAE models saved successfully.")

    @staticmethod
    def load(filepath: str):
        """Loads the encoder and decoder models from a directory."""
        if not os.path.isdir(filepath):
            print(f"Error: Directory not found at {filepath}. Cannot load models.")
            return None, None
            
        print(f"Loading VAE models from directory: {filepath}")
        try:
            encoder = tf.keras.models.load_model(os.path.join(filepath, 'encoder.keras'), custom_objects={'Sampling': Sampling})
            decoder = tf.keras.models.load_model(os.path.join(filepath, 'decoder.keras'))
            print("VAE models loaded successfully.")
            return encoder, decoder
        except Exception as e:
            print(f"An error occurred while loading the models: {e}")
            return None, None