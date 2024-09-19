import requests
import yfinance as yf
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from moexalgo import Ticker, session
from dotenv import load_dotenv
from scipy.stats import norm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats
import seaborn as sns
import os


load_dotenv()


def initialize_session(username, password):
    session.authorize(username, password)


def get_moex_info(tickers):
    """
    Получить рыночную капитализацию и секторы для каждого тикера.
    Возвращает общий список активов с рыночной капитализацией и секторами.
    """
    market_info = {}
    c = 0

    for ticker in tickers:
        try:
            # Получаем информацию с Yahoo Finance для тикера
            yf_ticker = yf.Ticker(f"{ticker}.ME")
            info = yf_ticker.info

            # Получаем рыночную капитализацию и сектор
            market_cap = info.get('marketCap', 0)
            sector = info.get('sector', 'Unknown')

            # Сохраняем информацию в словарь
            market_info[ticker] = {
                'marketCap': market_cap,
                'sector': sector
            }
        except Exception as e:
            print(f"Error fetching market cap or sector for {ticker}: {e}")
            market_info[ticker] = {
                'marketCap': 0,
                'sector': 'Unknown'
            }

        c += 1
        if c % 10 == 0:
            print(f"Processed {c} tickers")

    # Сортируем по рыночной капитализации
    sorted_market_caps = sorted(market_info.items(), key=lambda x: x[1]['marketCap'], reverse=True)

    return len(tickers), sorted_market_caps


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


def plot_asset_map_with_pareto(mean_returns, std_devs, pareto_optimal, most_preferred_var_index, output_dir='plots'):
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
    filepath = os.path.join(output_dir, "Asset_map.png")
    plt.savefig(filepath)
    print(f"График Asset Map сохранен в {filepath}")
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


def plot_acf_for_assets(log_returns, lags=40, output_dir='plots'):
    """
    Построить ACF график для каждого актива и сохранить в файл.
    """
    # Создаем директорию для сохранения графиков, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for ticker in log_returns.columns:
        returns = log_returns[ticker].dropna()  # Удаляем пустые значения
        if returns.empty or len(returns) < lags:
            print(f"Недостаточно данных для ACF графика для {ticker}. Пропущено.")
            continue

        plt.figure(figsize=(10, 6))
        plot_acf(returns, lags=lags, alpha=0.05)  # Построение ACF с указанным количеством лагов
        plt.title(f"ACF для {ticker}")

        # Сохраняем график в файл
        filepath = os.path.join(output_dir, f"ACF_{ticker}.png")
        plt.savefig(filepath)
        plt.close()  # Закрываем фигуру после сохранения, чтобы освободить память
        print(f"График ACF для {ticker} сохранен в {filepath}")


def test_white_noise(log_returns, max_lags=None):
    """
    Тест Льюнга-Бокса для проверки автокорреляции (белого шума) по каждому активу.
    """
    results = {}

    for ticker in log_returns.columns:
        returns = log_returns[ticker].dropna()  # Удаляем пустые значения для каждого актива
        if returns.empty or len(returns) < (max_lags or 10):  # Проверяем минимальное количество данных для теста
            print(f"Недостаточно данных для теста белого шума для {ticker}. Пропущено.")
            results[ticker] = "Недостаточно данных"
            continue

        # Определяем количество лагов: если не указано, берем √n наблюдений
        n_obs = len(returns)
        lags = max_lags if max_lags else int(np.sqrt(n_obs))  # Выбираем лаги, если не заданы
        lags = min(lags, len(returns) - 1)  # Убедимся, что лаги не превышают количество наблюдений
        print(ticker, lags)

        try:
            lb_test = acorr_ljungbox(returns, lags=[lags],
                                     return_df=True)  # Тест Льюнга-Бокса с заданным количеством лагов
            p_value = lb_test['lb_pvalue'].values[0]  # Извлекаем p-value
            if p_value > 0.05:
                results[ticker] = "Белый шум (случайность не отвергается)"
            else:
                results[ticker] = "Не белый шум (случайность отвергается)"
        except ValueError as e:
            print(f"Ошибка для {ticker}: {e}")
            results[ticker] = "Ошибка в данных"

    return results

def get_sectors(tickers):
    sectors = {}
    for ticker in tickers:
        try:
            yf_ticker = yf.Ticker(f"{ticker}.ME")
            info = yf_ticker.info
            sectors[ticker] = info.get('sector', 0)
        except Exception as e:
            print(f"Error fetching sector for {ticker}: {e}")
            sectors[ticker] = 0


def find_top_assets_by_sector(moex_info, n=5):
    """
    Возвращает первые n активов из разных секторов.
    """
    sectors_found = set()
    selected_assets = []

    for ticker, info in moex_info:
        sector = info['sector']
        if sector not in sectors_found:
            selected_assets.append((ticker, info))
            sectors_found.add(sector)
            if len(selected_assets) == n:
                break

    return selected_assets


def analyze_normality(log_returns, selected_assets):
    """
    Исследовать распределения доходностей выбранных активов на нормальность.
    Строит гистограммы, Q-Q графики и выполняет тесты на нормальность.
    """
    output_dir = 'Distribution_analysis'

    # Создаем директорию для сохранения графиков, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for asset, info in selected_assets:
        ticker = asset  # Извлекаем тикер актива из кортежа
        returns = log_returns[ticker].dropna()  # Удаляем пустые значения

        # Проверка на нормальность с помощью теста Шапиро-Уилка
        shapiro_test = stats.shapiro(returns)
        kstest = stats.kstest(returns, 'norm', args=(returns.mean(), returns.std()))

        # Строим гистограмму и Q-Q график
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(returns, kde=True, color='blue', bins=30)
        plt.title(f"Гистограмма распределения доходностей для {ticker}")
        plt.subplot(1, 2, 2)
        stats.probplot(returns, dist="norm", plot=plt)
        plt.title(f"Q-Q график для {ticker}")
        plt.savefig(f"{output_dir}/Distribution_{ticker}.png")
        plt.close()

        print(f"\nРаспределение доходностей для {ticker}:")
        print(f"Тест Шапиро-Уилка: p-value = {shapiro_test.pvalue:.5f}")
        print(f"Тест Колмогорова-Смирнова: p-value = {kstest.pvalue:.5f}")

if __name__ == "__main__":
    username = os.getenv('MOEX_USERNAME')
    password = os.getenv('MOEX_PASSWORD')

    initialize_session(username, password)

    tickers = get_moex_tickers()

    total_assets, moex_info = get_moex_info(tickers)
    main_companies = moex_info[:10]
    print(f"Total assets listed on MOEX: {total_assets}")
    print("Top 10 companies by market capitalization on MOEX:")
    for company, info in main_companies:
       print(f"{company}: Market Cap = {info['marketCap']}")

    price_data = fetch_data_for_tickers(tickers, start_date='2018-01-01', end_date='2018-12-31')

    log_returns = calculate_log_returns(price_data)

    mean_returns, std_devs = calculate_statistics(log_returns)

    pareto_optimal = find_pareto_optimal(mean_returns, std_devs)

    mean_returns_optimal, tickers_optimal, std_devs_optimal = optimal(pareto_optimal, mean_returns, tickers, std_devs)

    most_preferred_var_index, most_preferred_cvar_index = var_and_cvar_index(pareto_optimal, mean_returns)
    print(f"VaR most preferred: {tickers_optimal[most_preferred_var_index]}")
    print(f"CVaR most preferred: {tickers_optimal[most_preferred_cvar_index]}")
    plot_asset_map_with_pareto(mean_returns, std_devs, pareto_optimal, most_preferred_var_index)

    significant_assets = main_companies[:5]
    print("Significant companies by market capitalization on MOEX:")
    for company, market_cap in significant_assets:
        print(f"{company}: Market Cap = {market_cap}")
    selected_tickers = [ticker for ticker, _ in significant_assets]
    plot_acf_for_assets(log_returns[selected_tickers], lags=40)
    white_noise_results = test_white_noise(log_returns[selected_tickers], max_lags=None)
    for ticker, result in white_noise_results.items():
        print(f"{ticker}: {result}")

    selected_assets = find_top_assets_by_sector(moex_info)
    print(selected_assets)
    for asset, details in selected_assets:
        print(f"{asset}: Market Cap = {details['marketCap']}, Sector = {details['sector']}")
    analyze_normality(log_returns, selected_assets)


