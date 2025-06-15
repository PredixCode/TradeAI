import numpy as np
import pandas as pd
import os

from market.Stock import Stock
from market.util.FetchQueue import FetchQueue
from trade.Trader import Trader
from ai.TradeAI import TradeAI

# --- CONSTANT FOR MODEL DIRECTORY ---
MODEL_DIR = "ai/models/"

def get_state(data: pd.DataFrame, current_step: int, window_size: int):
    # ... (this function remains exactly the same)
    close_prices = data['Close'].values
    start = current_step - window_size + 1
    if start < 0:
        price_window = np.pad(close_prices[0:current_step + 1], (abs(start), 0), 'edge')
    else:
        price_window = close_prices[start:current_step + 1]
    normalized_prices = price_window / price_window[0] - 1
    timestamp = data.index[current_step]
    hour_sin = np.sin(2 * np.pi * timestamp.hour / 24)
    hour_cos = np.cos(2 * np.pi * timestamp.hour / 24)
    day_of_week_sin = np.sin(2 * np.pi * timestamp.dayofweek / 7)
    day_of_week_cos = np.cos(2 * np.pi * timestamp.dayofweek / 7)
    day_of_month_sin = np.sin(2 * np.pi * timestamp.day / 31)
    day_of_month_cos = np.cos(2 * np.pi * timestamp.day / 31)
    time_features = np.array([hour_sin, hour_cos, day_of_week_sin, day_of_week_cos, day_of_month_sin, day_of_month_cos])
    state = np.concatenate((normalized_prices, time_features))
    return np.reshape(state, [1, len(state)])

def initialize_agent(state_size: int) -> TradeAI:
    """Handles the interactive process of creating or loading an agent."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    saved_models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")]

    if not saved_models:
        print("No saved models found. Creating a new agent.")
        return TradeAI(state_size=state_size)

    while True:
        load_choice = input("Load a pre-trained model? (y/n): ").lower()
        if load_choice in ['y', 'n']:
            break
        print("Invalid input. Please enter 'y' or 'n'.")

    if load_choice == 'n':
        print("Creating a new agent.")
        return TradeAI(state_size=state_size)
    
    print("\nAvailable models:")
    for i, model_name in enumerate(saved_models):
        print(f"  {i + 1}: {model_name}")
    
    while True:
        try:
            model_choice = int(input("Enter the number of the model to load: "))
            if 1 <= model_choice <= len(saved_models):
                chosen_model_path = os.path.join(MODEL_DIR, saved_models[model_choice - 1])
                agent = TradeAI(state_size=state_size) # Create agent first
                agent.load(chosen_model_path) # Then load weights
                return agent
            else:
                print("Invalid number. Please choose from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    # --- 1. Setup ---
    stock_ticker = "NVDA"
    queue = FetchQueue()
    stock = Stock(stock_ticker)
    
    # --- 2. Data Preparation ---
    print("Fetching all high-frequency historical data for training...")
    data = stock.get_all_historical_data_accurate(queue=queue)
    
    if data.empty:
        raise SystemExit("Cannot run training, no data was fetched.")
    
    # --- 3. AI and Environment Initialization ---
    price_window_size = 60
    num_time_features = 6
    state_size = price_window_size + num_time_features
    
    # --- INTERACTIVE AGENT INITIALIZATION ---
    agent = initialize_agent(state_size)
    
    batch_size = 32
    num_episodes = 10

    # --- 4. The Training Loop ---
    for e in range(num_episodes):
        try:
            print(f"\n--- Starting Episode: {e+1}/{num_episodes} ---")
            trader = Trader(historical_data=data, initial_balance=10000.0)
            
            for step in range(trader.total_steps):
                state = get_state(data, step, price_window_size)
                action = agent.act(state)
                
                previous_portfolio_value = trader.portfolio_value
                if action == 1: trader.buy(amount_in_currency=1000)
                elif action == 2 and trader.shares_held > 0: trader.sell(amount_in_shares=trader.shares_held / 2)
                else: trader.hold()

                done = not trader.next_step()
                reward = trader.portfolio_value - previous_portfolio_value
                next_state = get_state(data, step + 1, price_window_size) if not done else state
                
                agent.remember(state, action, reward, next_state, done)
                agent.replay(batch_size)

                if done:
                    break
        except KeyboardInterrupt:
            break
        
        print(f"--- Episode {e+1} Summary ---")
        trader.print_summary()

    # --- 5. INTERACTIVE SAVE PROMPT ---
    while True:
        save_choice = input("\nDo you want to save the trained model? (y/n): ").lower()
        if save_choice in ['y', 'n']:
            break
        print("Invalid input. Please enter 'y' or 'n'.")

    if save_choice == 'y':
        default_filename = f"{stock_ticker}_{num_episodes}_episodes.keras"
        filename = input(f"Enter filename (default: {default_filename}): ")
        if not filename:
            filename = default_filename
        
        agent.save(os.path.join(MODEL_DIR, filename))

    print("\nApplication finished.")