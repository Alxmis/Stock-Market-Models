import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from moexalgo import Ticker, session
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize MOEX session
def initialize_session(username, password):
    session.authorize(username, password)

# Fetch general information about MOEX
def get_moex_info():
    url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
    response = requests.get(url)
    data = response.json()

    # Get the list of tickers (securities)
    securities_data = data['securities']['data']
    total_assets = len(securities_data)

    # Assuming 'list_level' contains the level of listing, 'SECNAME' contains the company name
    companies = [item[2] for item in securities_data[:10]]  # Extract names of the first 10 companies (for example)

    return total_assets, companies

# Fetch tickers from MOEX
def get_moex_tickers():
    url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
    response = requests.get(url)
    data = response.json()
    tickers = [item[0] for item in data['securities']['data']]  # Adjust indexing as needed
    return tickers

# Fetch historical price data for each ticker using moexalgo
def fetch_data_for_tickers(tickers, start_date='2018-01-01', end_date='2018-12-31', period=24):
    price_data_list = []
    ticker_names = []

    c = 0
    for ticker in tickers:
        if c == 10:
            break
        try:
            moex_ticker = Ticker(ticker)
            candles = moex_ticker.candles(start=start_date, end=end_date, period=period)

            if candles.empty:
                print(f"No data available for {ticker} in the specified date range.")
            else:
                price_data_list.append(candles['close'].reset_index(drop=True))
                ticker_names.append(ticker)
                print(f"Data for {ticker} fetched successfully.")
                c += 1
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    price_data = pd.concat(price_data_list, axis=1)
    price_data.columns = ticker_names
    return price_data

# Calculate log returns
def calculate_log_returns(price_data):
    if price_data.empty:
        print("No price data available for any tickers.")
        return pd.DataFrame()

    log_returns = np.log(price_data / price_data.shift(1))
    log_returns.dropna(inplace=True)
    return log_returns

# Calculate mean returns and standard deviations
def calculate_statistics(log_returns):
    if log_returns.empty:
        print("No log returns available for calculation.")
        return pd.Series(), pd.Series()

    mean_returns = log_returns.mean()
    std_devs = log_returns.std()
    return mean_returns, std_devs

# Find Pareto-optimal assets
def find_pareto_optimal(mean_returns, std_devs):
    is_pareto_optimal = np.ones(mean_returns.shape[0], dtype=bool)
    for i, (mean_i, std_i) in enumerate(zip(mean_returns, std_devs)):
        for j, (mean_j, std_j) in enumerate(zip(mean_returns, std_devs)):
            if i != j:
                if (mean_j > mean_i) and (std_j < std_i):
                    is_pareto_optimal[i] = False
                    break
    return is_pareto_optimal

# Plot asset map (σ, E) and highlight Pareto-optimal assets
def plot_asset_map_with_pareto(mean_returns, std_devs, pareto_optimal):
    if mean_returns.empty or std_devs.empty:
        print("Insufficient data for plotting the asset map.")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(std_devs, mean_returns, c='blue', marker='o', label="Assets")

    # Highlight Pareto-optimal assets
    plt.scatter(std_devs[pareto_optimal], mean_returns[pareto_optimal], c='red', marker='o', label="Pareto Optimal")

    for i, ticker in enumerate(mean_returns.index):
        plt.annotate(ticker, (std_devs[i], mean_returns[i]), fontsize=12)

    plt.title('Asset Map (σ, E) with Pareto Optimal Assets')
    plt.xlabel('Standard Deviation (Risk)')
    plt.ylabel('Mean Return')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
if __name__ == "__main__":
    # Fetch MOEX credentials from environment variables
    username = os.getenv('MOEX_USERNAME')
    password = os.getenv('MOEX_PASSWORD')

    # Initialize session
    initialize_session(username, password)

    # Get general MOEX information
    total_assets, main_companies = get_moex_info()
    print(f"Total assets listed on MOEX: {total_assets}")
    print("Main companies listed on MOEX:", ", ".join(main_companies))

    # Get list of tickers
    tickers = get_moex_tickers()

    # Fetch price data for 2018
    price_data = fetch_data_for_tickers(tickers, start_date='2018-01-01', end_date='2018-12-31')

    # Calculate log returns
    log_returns = calculate_log_returns(price_data)

    # Calculate mean returns and standard deviations
    mean_returns, std_devs = calculate_statistics(log_returns)

    # Find Pareto-optimal assets
    pareto_optimal = find_pareto_optimal(mean_returns, std_devs)

    # Plot asset map with Pareto-optimal assets highlighted
    plot_asset_map_with_pareto(mean_returns, std_devs, pareto_optimal)
