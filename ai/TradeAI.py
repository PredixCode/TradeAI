import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class TradeAI:
    """
    A Deep Q-Network (DQN) agent designed to learn trading strategies.
    """
    def __init__(self, state_size: int, action_size: int = 3):
        """
        Initializes the DQN Agent.

        Args:
            state_size (int): The number of features describing the market state.
            action_size (int): The number of possible actions (default is 3: hold, buy, sell).
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Experience Replay memory
        self.memory = deque(maxlen=2000)
        
        # --- Hyperparameters ---
        self.gamma = 0.95   # Discount rate for future rewards
        self.epsilon = 1.0  # Exploration rate (starts at 100%)
        self.epsilon_min = 0.01 # Minimum exploration rate
        self.epsilon_decay = 0.995 # Rate at which exploration decreases
        self.learning_rate = 0.001
        
        self.model = self._build_model()
        print("TradeAI (DQN Agent) initialized.")

    def _build_model(self) -> Sequential:
        """Builds the neural network model for the Q-function."""
        model = Sequential()
        # Input layer's shape is the state size
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        # Output layer has one neuron for each possible action
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Stores an experience tuple in the memory buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state) -> int:
        """
        Chooses an action based on the current state (Epsilon-Greedy policy).

        Args:
            state: The current market state.

        Returns:
            int: The action to take (0: hold, 1: buy, 2: sell).
        """
        # Exploration: Take a random action
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Exploitation: Ask the model for the best action
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # Returns the index of the highest Q-value

    def replay(self, batch_size: int):
        """Trains the neural network using a random batch of experiences."""
        if len(self.memory) < batch_size:
            return # Not enough memories to train yet

        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            # The target Q-value is the immediate reward
            target = reward
            if not done:
                # If the episode isn't over, add the discounted future reward
                # We predict the Q-values for the next state and take the max
                future_rewards = self.model.predict(next_state, verbose=0)[0]
                target = reward + self.gamma * np.amax(future_rewards)
            
            # Get the model's current prediction for the Q-values of the original state
            target_f = self.model.predict(state, verbose=0)
            # Update only the Q-value for the action we actually took
            target_f[0][action] = target
            
            # Train the model on this one corrected experience
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        # Decay the exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay