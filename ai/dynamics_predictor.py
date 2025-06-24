import tensorflow as tf
from tensorflow.keras import layers, Model
import os

class DynamicsPredictorLSTM(Model):
    """
    An LSTM-based model designed to predict the next latent state vector (z)
    in a sequence, learning the temporal dynamics of the market.
    """
    def __init__(self, latent_dim: int, sequence_length: int, **kwargs):
        super(DynamicsPredictorLSTM, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        
        self.model = self._build_model()
        print(f"DynamicsPredictorLSTM initialized. Latent Dim: {latent_dim}, Sequence Length: {sequence_length}")

    def _build_model(self) -> Model:
        """Builds the LSTM network."""
        model_input = layers.Input(shape=(self.sequence_length, self.latent_dim))
        
        # A stack of LSTM layers can learn more complex patterns.
        # return_sequences=True passes the output of each timestep to the next layer.
        x = layers.LSTM(128, return_sequences=True)(model_input)
        x = layers.Dropout(0.2)(x)
        
        # The final LSTM layer only needs to return the output of the last timestep.
        x = layers.LSTM(128, return_sequences=False)(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        
        # The output layer predicts the next z_vector, so its size must be latent_dim.
        model_output = layers.Dense(self.latent_dim, activation='linear')(x) # Linear activation for regression
        
        model = Model(model_input, model_output, name="dynamics_predictor")
        print("Dynamics Predictor (LSTM) model built successfully.")
        model.summary()
        return model

    def call(self, inputs):
        """Defines the forward pass."""
        return self.model(inputs)

    def save(self, filepath: str):
        """Saves the LSTM model."""
        print(f"Saving Dynamics Predictor model to {filepath}...")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print("Model saved successfully.")

    @staticmethod
    def load(filepath: str):
        """Loads a pre-trained LSTM model."""
        if not os.path.exists(filepath):
            print(f"Warning: Model file not found at {filepath}. Predictor will not be loaded.")
            return None
        
        print(f"Loading Dynamics Predictor model from {filepath}...")
        try:
            model = tf.keras.models.load_model(filepath)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            return None