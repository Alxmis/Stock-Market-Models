import matplotlib.pyplot as plt


def showing_graph_9(stock_Name, day, stock_returns):
    plt.plot(day, stock_returns, linestyle='-')
    plt.xlabel('День')
    plt.ylabel('Стоимость')
    plt.title(f'Динамика стоимости акции {stock_Name}')
    plt.show()


stock_Name = 'Лукойл'
stock_returns = log_returns[stock_Name] 

day = list(range(360))  

showing_graph_9(stock_Name, day, stock_returns)
