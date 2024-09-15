import numpy as np
import pandas as pd
from scipy.stats import norm

#функция для расчета Var с уровнем доверия=0.95 
def calculate_var(returns, confidence_level=0.95):
    """
    returns (pandas.Series)- значения VaR для каждого актива.
    """
    var = -returns.quantile(1 - confidence_level)
    return var

#рассчитывает CVaR для заданного набора доходностей
def calculate_cvar(returns, confidence_level=0.95):
    """
    returns pandas.Series - значения CVaR для каждого актива.
    """
    var = calculate_var(returns, confidence_level)
    cvar = -returns[returns <= -var].mean()
    return cvar

pareto_profit=[]

for i in range (0, len(pareto_optimal)):
    if (pareto_optimal[i]==True):
        pareto_profit.append(log_returns[i])
        #pareto_profit.append(log_returns.iloc[:, i])


var = calculate_var(pareto_profit, 0.95)
cvar = calculate_cvar(pareto_profit, 0.95)

# Определяем наиболее предпочтительный актив по VaR и CVaR
most_preferred_var = pareto_returns.columns[np.argmin(var)]
most_preferred_cvar = pareto_returns.columns[np.argmin(cvar)]

print(f"Наиболее предпочтительный актив по VaR: {most_preferred_var}")
print(f"Наиболее предпочтительный актив по CVaR: {most_preferred_cvar}")