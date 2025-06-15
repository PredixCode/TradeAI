import numpy as np
import pandas as pd

from market.Stock import Stock
from market.util.FetchQueue import FetchQueue
from trade.Trader import Trader
from ai.TradeAI import TradeAI



def get_state(data: pd.DataFrame, current_step: int, window_size: int):
    """
    Creates a rich, time-aware state representation for the AI.

    The state consists of two parts:
    1. A window of recent, normalized closing prices.
    2. A set of engineered time features from the current timestamp.

    Args:
        data (pd.DataFrame): The full historical data DataFrame.
        current_step (int): The current index in the DataFrame.
        window_size (int): The number of past prices to include in the state.

    Returns:
        np.array: A flattened numpy array representing the complete state.
    """
    # --- Part 1: Price Window ---
    close_prices = data['Close'].values
    start = current_step - window_size + 1
    
    if start < 0:
        price_window = np.pad(close_prices[0:current_step + 1], (abs(start), 0), 'edge')
    else:
        price_window = close_prices[start:current_step + 1]
    
    # Normalize the price window
    normalized_prices = price_window / price_window[0] - 1

    # --- Part 2: Time Feature Engineering ---
    timestamp = data.index[current_step]
    
    # We use sin/cos to represent cyclical features (like hours on a clock)
    # This helps the model understand that 23:00 is close to 00:00
    hour_sin = np.sin(2 * np.pi * timestamp.hour / 24)
    hour_cos = np.cos(2 * np.pi * timestamp.hour / 24)
    
    day_of_week_sin = np.sin(2 * np.pi * timestamp.dayofweek / 7)
    day_of_week_cos = np.cos(2 * np.pi * timestamp.dayofweek / 7)
    
    day_of_month_sin = np.sin(2 * np.pi * timestamp.day / 31)
    day_of_month_cos = np.cos(2 * np.pi * timestamp.day / 31)

    # Combine all time features into a single array
    time_features = np.array([
        hour_sin, hour_cos, 
        day_of_week_sin, day_of_week_cos,
        day_of_month_sin, day_of_month_cos
    ])

    # --- Part 3: Combine and Return ---
    # Concatenate the price features and time features into one state vector
    state = np.concatenate((normalized_prices, time_features))
    
    return np.reshape(state, [1, len(state)])


from market.Stock import Stock
from market.util.FetchQueue import FetchQueue
from trade.Trader import Trader
from ai.TradeAI import TradeAI

if __name__ == "__main__":
    # --- 1. Setup ---
    queue = FetchQueue()
    stock = Stock("NVDA")
    
    # --- 2. Data Preparation ---
    print("Fetching all high-frequency historical data for training...")
    data = stock.get_all_historical_data_accurate(queue=queue)
    
    if data.empty:
        raise SystemExit("Cannot run training, no data was fetched.")
    
    # --- 3. AI and Environment Initialization ---
    price_window_size = 60  # The AI will look at the last 30 data points (e.g., 30 minutes)
    num_time_features = 6   # We created 6 time features (sin/cos for hour, day_week, day_month)
    
    # The state size is now the sum of the price window and our time features
    state_size = price_window_size + num_time_features
    
    agent = TradeAI(state_size=state_size)
    trader = Trader(historical_data=data, initial_balance=10000.0)
    
    batch_size = 32
    num_episodes = 10

    # --- 4. The Training Loop (now more powerful) ---
    for e in range(num_episodes):
        print(f"\n--- Starting Episode: {e+1}/{num_episodes} ---")
        trader = Trader(historical_data=data, initial_balance=10000.0)
        
        # The loop now iterates over every single data point, regardless of time step
        for step in range(trader.total_steps):
            # a. Get the current time-aware state
            state = get_state(data, step, price_window_size)
            
            # b. Agent chooses an action
            action = agent.act(state)
            
            # c. Execute action (logic remains the same)
            previous_portfolio_value = trader.portfolio_value
            if action == 1: trader.buy(amount_in_currency=1000)
            elif action == 2 and trader.shares_held > 0: trader.sell(amount_in_shares=trader.shares_held / 2)
            else: trader.hold()

            # d. Advance environment and get reward
            done = not trader.next_step()
            reward = trader.portfolio_value - previous_portfolio_value
            
            # e. Get the next time-aware state
            next_state = get_state(data, step + 1, price_window_size) if not done else state
            
            # f. Agent remembers and replays (logic remains the same)
            agent.remember(state, action, reward, next_state, done)
            agent.replay(batch_size)

            if done:
                break
        
        print(f"--- Episode {e+1} Summary ---")
        trader.print_summary()