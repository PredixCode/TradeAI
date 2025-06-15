import os
import numpy as np
import pandas as pd
import threading
import time
import tensorflow as tf

from market.Stock import Stock
from market.util.FetchQueue import FetchQueue
from trade.Trader import Trader
from ai.TradeAI import TradeAI

# --- CONSTANT FOR MODEL DIRECTORY ---
MODEL_DIR = "ai/models/"

def get_state(data: pd.DataFrame, trader: Trader, current_step: int, window_size: int):
    """
    Creates a rich, time-aware and portfolio-aware state representation for the AI.
    """
    # --- Part 1: Price Window ---
    close_prices = data['Close'].values
    start = current_step - window_size + 1
    if start < 0:
        price_window = np.pad(close_prices[0:current_step + 1], (abs(start), 0), 'edge')
    else:
        price_window = close_prices[start:current_step + 1]
    normalized_prices = price_window / price_window[0] - 1

    # --- Part 2: Time Features ---
    timestamp = data.index[current_step]
    hour_sin = np.sin(2 * np.pi * timestamp.hour / 24)
    hour_cos = np.cos(2 * np.pi * timestamp.hour / 24)
    day_of_week_sin = np.sin(2 * np.pi * timestamp.dayofweek / 7)
    day_of_week_cos = np.cos(2 * np.pi * timestamp.dayofweek / 7)
    time_features = np.array([hour_sin, hour_cos, day_of_week_sin, day_of_week_cos])

    # --- Part 3: Portfolio Features ---
    # Normalize balance and shares held to be between 0 and 1
    normalized_balance = trader.current_balance / trader.portfolio_value if trader.portfolio_value != 0 else 0
    normalized_shares_value = (trader.shares_held * data['Close'].iloc[current_step]) / trader.portfolio_value if trader.portfolio_value != 0 else 0
    portfolio_features = np.array([normalized_balance, normalized_shares_value])

    # --- Part 4: Combine and Return ---
    state = np.concatenate((normalized_prices, time_features, portfolio_features))
    return np.reshape(state, [1, len(state)])

def actor_worker(agent, data, stop_event, episode_counter):
    """A worker thread that runs simulations to generate experiences."""
    print(f"[{threading.current_thread().name}] Actor starting...")
    while not stop_event.is_set():
        episode_num = next(episode_counter)
        trader = Trader(historical_data=data, initial_balance=10000.0)
        
        for step in range(trader.total_steps):
            if stop_event.is_set(): break

            state = get_state(data, trader, step, price_window_size)
            action = agent.act(state)
            
            # Simplified trading logic for clarity
            previous_portfolio_value = trader.portfolio_value
            trade_executed = False
            if action == 0: trade_executed = trader.hold()
            elif 1 <= action <= 3:
                amount_to_buy = trader.current_balance * {1: 0.05, 2: 0.10, 3: 0.15}[action]
                trade_executed = trader.buy(amount_to_buy)
            elif 4 <= action <= 6:
                amount_to_sell = trader.shares_held * {4: 0.25, 5: 0.50, 6: 1.0}[action]
                trade_executed = trader.sell(amount_to_sell)

            reward = trader.portfolio_value - previous_portfolio_value if trade_executed else -1
            done = not trader.next_step()
            next_state = get_state(data, trader, step + 1, price_window_size) if not done else state
            
            agent.remember(state, action, reward, next_state, done)

            if done: break
        
        print(f"[{threading.current_thread().name}] Episode {episode_num} finished. Final Value: €{trader.portfolio_value:,.2f}")

def learner_worker(agent, batch_size, stop_event):
    """A worker thread that continuously trains the agent's model."""
    print(f"[{threading.current_thread().name}] Learner starting...")
    while not stop_event.is_set():
        agent.replay(batch_size)
        time.sleep(0.1) # Small sleep to prevent busy-waiting if memory is empty

def initialize_agent(state_size: int, action_size: int) -> TradeAI:
    """Handles the interactive process of creating or loading an agent."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    saved_models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")]

    if not saved_models:
        print("No saved models found. Creating a new agent.")
        return TradeAI(state_size=state_size, action_size=action_size)

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

def gpu_check():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"✅ Found and configured {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        print("❌ No GPU found. Training will proceed on CPU.")


if __name__ == "__main__":
    gpu_check()
    
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
    price_window_size = 30
    state_size = price_window_size + 4 + 2 # price_window + time_features + portfolio_features
    action_size = 7 
    batch_size = 64
    
    agent = initialize_agent(state_size, action_size)

    MAX_STUCK_STEPS = 10
    PUNISHMENT_SCALER = 0.1

    # --- 4. The Continuous Training Loop ---
    episode = 0
    try:
        while True:
            episode += 1
            print(f"\n--- Starting Episode: {episode} ---")
            trader = Trader(historical_data=data, initial_balance=10000.00)
            
            stuck_counter = 0
            total_rewards = 0
            
            for step in range(trader.total_steps):
                # Get the current state and action BEFORE checking for failure conditions
                state = get_state(data, trader, step, price_window_size)
                action = agent.act(state)

                # --- Early Termination Punishment ---
                if trader.is_insolvent or stuck_counter >= MAX_STUCK_STEPS:
                    if trader.is_insolvent:
                        print(f"  -> BANKRUPTCY! Agent is insolvent at step {step}. Ending episode.")
                    else:
                        print(f"  -> STUCK! Agent failed to execute a valid trade for {MAX_STUCK_STEPS} steps. Ending episode.")
                    
                    # Calculate the linear punishment
                    remaining_steps = trader.total_steps - step
                    punishment = -remaining_steps * PUNISHMENT_SCALER
                    total_rewards += punishment
                    
                    print(f"  -> Survival Penalty: {punishment:.2f} (for missing {remaining_steps} steps)")
                    
                    # Remember this final, catastrophic failure and mark the episode as done
                    agent.remember(state, action, punishment, state, True)
                    break # End the episode

                # --- Original Trading Logic ---
                previous_portfolio_value = trader.portfolio_value
                
                trade_executed = False
                if action == 0: # Hold
                    trade_executed = trader.hold()
                elif 1 <= action <= 3: # Buy Actions
                    buy_fractions = {1: 0.005, 2: 0.01, 3: 0.075} 
                    amount_to_buy = trader.current_balance * buy_fractions[action]
                    trade_executed = trader.buy(amount_to_buy)
                elif 4 <= action <= 6: # Sell Actions
                    if trader.shares_held > 0:
                        sell_fractions = {4: 0.25, 5: 0.50, 6: 1.0}
                        amount_to_sell = trader.shares_held * sell_fractions[action]
                        trade_executed = trader.sell(amount_to_sell)
                    else:
                        trade_executed = False

                if trade_executed:
                    stuck_counter = 0
                    reward = trader.portfolio_value - previous_portfolio_value
                else:
                    stuck_counter += 1
                    reward = -100

                done = not trader.next_step()
                
                next_state = get_state(data, trader, step + 1, price_window_size) if not done else state
                
                agent.remember(state, action, reward, next_state, done)
                agent.replay(batch_size)

                total_rewards += reward
                if done:
                    break
            
            print(f"--- Episode {episode} Summary ---")
            trader.print_summary()
            print(f"Agent Reward Balance: {total_rewards:.4f}")
            print(f"Agent Epsilon: {agent.epsilon:.4f}")

    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted by user. Stopping...")

    # --- 5. SAVE PROMPT ---
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