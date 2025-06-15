import numpy as np
import random
import os
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

class TradeAI:
    """
    A Deep Q-Network (DQN) agent designed to learn trading strategies.
    """
    def __init__(self, state_size: int, action_size: int = 3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        print("TradeAI (DQN Agent) initialized.")

    def _build_model(self) -> Sequential:
        """Builds the neural network model for the Q-function."""
        # Using the modern Keras API with an explicit Input layer
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(24, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def save(self, filepath: str):
        """Saves the current model's weights and architecture."""
        print(f"Saving model to {filepath}...")
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print("Model saved successfully.")

    def load(self, filepath: str):
        """Loads a pre-trained model from a file."""
        if not os.path.exists(filepath):
            print(f"Warning: Model file not found at {filepath}. Agent will not be loaded.")
            return
        
        print(f"Loading model from {filepath}...")
        self.model = load_model(filepath)
        # When loading a trained model, we want it to exploit its knowledge,
        # so we set the exploration rate to its minimum value.
        self.epsilon = self.epsilon_min
        print("Model loaded successfully.")

    def remember(self, state, action, reward, next_state, done):
        """Stores an experience tuple in the memory buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state) -> int:
        """Chooses an action based on the current state (Epsilon-Greedy policy)."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size: int):
        """
        Trains the neural network using a random batch of experiences.
        """
        if len(self.memory) < batch_size:
            return # Not enough memories to train yet

        # 1. Sample a minibatch from memory
        minibatch = random.sample(self.memory, batch_size)

        # 2. Extract states, actions, rewards, etc., into separate NumPy arrays
        # This is much more efficient for batch processing.
        states = np.array([experience[0] for experience in minibatch]).reshape(batch_size, self.state_size)
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch]).reshape(batch_size, self.state_size)
        dones = np.array([experience[4] for experience in minibatch])

        # 3. Predict Q-values for the current states and next states IN A SINGLE BATCH
        # This replaces 2 * batch_size individual predict calls with just 2 calls.
        q_values_current = self.model.predict(states, verbose=0)
        q_values_next = self.model.predict(next_states, verbose=0)

        # 4. Calculate the target Q-values for the entire batch
        # Start with the current Q-values as a base
        targets = q_values_current.copy()
        
        # The target for each experience is its immediate reward
        # For experiences that are not 'done', add the discounted maximum future Q-value
        # This is the core of the Bellman equation in Q-learning
        targets[np.arange(batch_size), actions] = rewards + self.gamma * np.amax(q_values_next, axis=1) * (1 - dones)

        # 5. Train the model on the entire batch in a single call
        # This replaces batch_size individual fit calls with just ONE call.
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        # 6. Decay the exploration rate (epsilon)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay