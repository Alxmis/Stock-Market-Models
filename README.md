# 📈 Financial Asset Analysis: VaR, CVaR, and Pareto Optimization

This project provides tools for analyzing financial assets, focusing on risk assessment and return optimization using techniques such as Value at Risk (VaR), Conditional Value at Risk (CVaR), and Pareto optimization. Additionally, the project allows the investigation of whether the returns of financial assets follow specific distributions, and provides graphical insights into the relationship between risk and return.

## 🚀 Key Features

- **Market Data Collection**: Fetch historical price data from Yahoo Finance and MOEX using convenient API integrations.
- **Logarithmic Returns Calculation**: Compute log returns for financial assets to analyze their historical performance.
- **Risk and Return Analysis**:
  - **VaR (Value at Risk)**: Estimate the potential maximum loss of assets over a specific period with a given confidence level.
  - **CVaR (Conditional Value at Risk)**: Assess the average loss in worst-case scenarios beyond VaR.
  - **Pareto Optimal Asset Selection**: Identify assets that offer the best trade-off between return and risk.
- **Distribution Fitting**: Automatically fit asset returns to various probability distributions using `distfit` and assess their conformity.
- **Visualizations**: Plot risk-return scatterplots, ACF (autocorrelation) plots, and distribution histograms with Q-Q plots.

## 🛠 Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/financial-asset-analysis.git
cd financial-asset-analysis
pip install -r requirements.txt
```

## 📋 Usage

### 1. Set up API Access

Create a .env file to store your credentials for accessing the MOEX API:

```bash
MOEX_USERNAME=your_username
MOEX_PASSWORD=your_password
```

### 2. Fetch Data

Run the script to fetch and process asset data from MOEX or Yahoo Finance:

```bash
python main.py
```
