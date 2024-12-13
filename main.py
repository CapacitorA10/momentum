# main.py
from strategy_modules import *
import pandas as pd

if __name__ == "__main__":
    tickers = load_universe()
    adj_close = load_price_data(tickers)
    fundamental_data = load_fundamental_data(tickers)

    all_dates = adj_close.index
    quarter_ends = all_dates[all_dates.is_quarter_end]
    rebalance_dates = [all_dates[all_dates >= (qe + pd.DateOffset(months=1))][0] for qe in quarter_ends if len(all_dates[all_dates >= (qe + pd.DateOffset(months=1))]) > 0]

    result = backtest(adj_close, fundamental_data, rebalance_dates)

    final_value = result['portfolio_value'].iloc[-1]
    total_return = final_value / result['portfolio_value'].iloc[0] - 1
    annualized_return = (1 + total_return) ** (252 / len(result)) - 1

    print("Final Portfolio Value:", final_value)
    print("Total Return:", total_return)
    print("Annualized Return:", annualized_return)
