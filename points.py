import requests
import yfinance as yf
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from moexalgo import Ticker, session
from dotenv import load_dotenv
from scipy.stats import norm
import os


load_dotenv()


def initialize_session(username, password):
    session.authorize(username, password)


def get_moex_info(tickers):
    market_caps = {}
    for ticker in tickers:
        try:
            yf_ticker = yf.Ticker(f"{ticker}.ME")  # Adjusted for Yahoo Finance's Russian tickers
            info = yf_ticker.info
            market_caps[ticker] = info.get('marketCap', 0)
        except Exception as e:
            print(f"Error fetching market cap for {ticker}: {e}")
            market_caps[ticker] = 0  # Set 0 if market cap is not available

    sorted_market_caps = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)

    top_10_companies = sorted_market_caps[:10]

    return len(tickers), top_10_companies


def get_moex_tickers():
    url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
    response = requests.get(url)
    data = response.json()
    tickers = [item[0] for item in data['securities']['data']]  # Extract tickers
    return tickers


def fetch_data_for_tickers(tickers, start_date='2018-01-01', end_date='2018-12-31', period=24):
    price_data_list = []
    ticker_names = []

    for ticker in tickers:
        try:
            moex_ticker = Ticker(ticker)
            candles = moex_ticker.candles(start=start_date, end=end_date, period=period)

            if candles.empty:
                print(f"No data available for {ticker} in the specified date range.")
            else:
                price_data_list.append(candles['close'].reset_index(drop=True))
                ticker_names.append(ticker)
                print(f"Data for {ticker} fetched successfully.")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    price_data = pd.concat(price_data_list, axis=1)
    price_data.columns = ticker_names
    return price_data


def calculate_log_returns(price_data):
    if price_data.empty:
        print("No price data available for any tickers.")
        return pd.DataFrame()

    log_returns = np.log(price_data / price_data.shift(1))
    # print(price_data)
    # print(log_returns)
    # TODO: use dropna or not? help
    # log_returns.dropna(inplace=True)
    # print(log_returns)
    return log_returns


def calculate_statistics(log_returns):
    if log_returns.empty:
        print("No log returns available for calculation.")
        return pd.Series(), pd.Series()

    mean_returns = log_returns.mean()
    std_devs = log_returns.std()
    return mean_returns, std_devs


def find_pareto_optimal(mean_returns, std_devs):
    is_pareto_optimal = np.ones(mean_returns.shape[0], dtype=bool)
    for i, (mean_i, std_i) in enumerate(zip(mean_returns, std_devs)):
        for j, (mean_j, std_j) in enumerate(zip(mean_returns, std_devs)):
            if i != j:
                if (mean_j > mean_i) and (std_j < std_i):
                    is_pareto_optimal[i] = False
                    break
    return is_pareto_optimal


def plot_asset_map_with_pareto(mean_returns, std_devs, pareto_optimal, most_preferred_var_index):
    if mean_returns.empty or std_devs.empty:
        print("Insufficient data for plotting the asset map.")
        return

    plt.figure(figsize=(18, 10))
    plt.scatter(std_devs, mean_returns, c='blue', marker='o', label="Assets", alpha=0.7)  # Add transparency to non-Pareto points

    # Highlight Pareto-optimal assets
    plt.scatter(std_devs[pareto_optimal], mean_returns[pareto_optimal], c='red', marker='o', label="Pareto Optimal", alpha=0.9)

    plt.scatter(std_devs_optimal[most_preferred_var_index], mean_returns_optimal[most_preferred_var_index], c='green', marker='o', label="Most Preferred", alpha=0.9)

    # Annotate only Pareto-optimal assets to reduce clutter
    for i, ticker in enumerate(mean_returns.index):
        if pareto_optimal[i]:
            plt.annotate(ticker, (std_devs[i], mean_returns[i]), fontsize=12)  # Adjust font size for Pareto-optimal assets

    # Zoom in on the relevant part of the plot, adjust as needed
    plt.xlim(0, 0.1)  # Adjust the range of x-axis (Standard Deviation)
    plt.ylim(-0.005, 0.01)  # Adjust the range of y-axis (Mean Return)

    plt.title('Asset Map (σ, E) with Pareto Optimal Assets')
    plt.xlabel('Standard Deviation (Risk)')
    plt.ylabel('Mean Return')
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_var(returns, confidence_level=0.95):
    return -np.percentile(returns, (1 - confidence_level) * 100)


def calculate_cvar(returns, confidence_level=0.95):
    return -np.mean(returns[returns <= -calculate_var(returns, confidence_level)])

def optimal(pareto_optimal, mean_returns, tickers, std_devs):
    mean_returns_optimal = []
    tickers_optimal = []
    std_devs_optimal = []
    for i in range(0, len(pareto_optimal)):
        if pareto_optimal[i] and not np.isnan(mean_returns[i]):
            mean_returns_optimal.append(mean_returns[i])
            tickers_optimal.append(tickers[i])
            std_devs_optimal.append(std_devs[i])

    return mean_returns_optimal, tickers_optimal, std_devs_optimal


def var_and_cvar_index(pareto_optimal, mean_returns):
    var = []
    cvar = []

    for i in range(0, len(pareto_optimal)):
        if pareto_optimal[i] and not np.isnan(mean_returns[i]):
            var.append(calculate_var(mean_returns[i], 0.95))
            cvar.append(calculate_cvar(mean_returns[i], 0.95))

    var = np.array(var)
    cvar = np.array(cvar)

    return np.argmin(var), np.argmin(cvar)


if __name__ == "__main__":
    username = os.getenv('MOEX_USERNAME')
    password = os.getenv('MOEX_PASSWORD')

    initialize_session(username, password)

    tickers = get_moex_tickers()

    total_assets, main_companies = get_moex_info(tickers)
    #print(f"Total assets listed on MOEX: {total_assets}")
    #print("Top 10 companies by market capitalization on MOEX:")
    #for company, market_cap in main_companies:
    #    print(f"{company}: Market Cap = {market_cap}")

    price_data = fetch_data_for_tickers(tickers, start_date='2018-01-01', end_date='2018-12-31')

    log_returns = calculate_log_returns(price_data)

    mean_returns, std_devs = calculate_statistics(log_returns)

    pareto_optimal = find_pareto_optimal(mean_returns, std_devs)

    mean_returns_optimal, tickers_optimal, std_devs_optimal = optimal(pareto_optimal, mean_returns, tickers, std_devs)

    most_preferred_var_index, most_preferred_cvar_index = var_and_cvar_index(pareto_optimal, mean_returns)

    print(f"VaR most preferred: {tickers_optimal[most_preferred_var_index]}")
    print(f"CVaR most preferred: {tickers_optimal[most_preferred_cvar_index]}")

    plot_asset_map_with_pareto(mean_returns, std_devs, pareto_optimal, most_preferred_var_index)