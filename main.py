from market.Stock import Stock
from trade.Trader import Trader

if __name__ == "__main__":
    # --- Setup ---
    stock = Stock("TSLA")

    # --- Data Preparation ---
    print("Fetching historical data for the trading simulation...")
    historical_data = stock.get_historical_data(period="max", interval="1m")

    if historical_data.empty:
        print("Cannot run simulation, no data was fetched.")
    else:
        # --- Simulation ---
        trader = Trader(historical_data=historical_data, initial_balance=10000.0)

        # Simple, rule-based strategy:
        # - Buy €2,000 worth on step 5
        # - Buy another €3,000 worth on step 20
        # - Sell all shares held on step 40
        
        simulation_running = True
        while simulation_running:
            # --- Your Strategy Logic ---
            if trader.current_step == 5:
                trader.buy(amount_in_currency=2000)
            
            elif trader.current_step == 20:
                trader.buy(amount_in_currency=3000)

            elif trader.current_step == 40:
                trader.sell(amount_in_shares=trader.shares_held)
            
            else:
                trader.hold()

            simulation_running = trader.next_step()

        # --- Results ---
        trader.print_summary()