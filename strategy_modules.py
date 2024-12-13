# strategy_modules.py
import pandas as pd
import numpy as np
import yfinance as yf
import cvxpy as cp
from config import *


# 데이터 로더 및 전처리
def load_universe():
    df = pd.read_csv(KOSPI200_LIST_PATH)
    return df['ticker'].tolist()


def load_price_data(tickers):
    price_data = yf.download(tickers, start=START_DATE, end=END_DATE)
    adj_close = price_data['Adj Close']
    return adj_close


def load_fundamental_data(tickers):
    fundamentals = {}
    for t in tickers:
        try:
            df = pd.read_csv(f'./data/fundamentals/{t}_fundamentals.csv', parse_dates=['date'])
            df.set_index('date', inplace=True)
            fundamentals[t] = df
        except FileNotFoundError:
            continue
    return fundamentals


# 팩터 계산
def calculate_rsi(prices, period=RSI_PERIOD):
    delta = prices.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rsi = 100 - (100 / (1 + up / down))
    return rsi


def geometric_growth_rate(series, periods=4):
    if len(series) < periods:
        return np.nan
    growth = (series.iloc[-1] / series.iloc[0]) ** (1 / periods) - 1
    return growth


def calculate_factors(price_data, fundamental_data):
    latest_prices = price_data.iloc[-200:]
    rsi_df = latest_prices.apply(calculate_rsi)
    latest_rsi = rsi_df.iloc[-1]

    revenue_growth, op_income_growth, roe_values = {}, {}, {}

    for t, df in fundamental_data.items():
        df = df.sort_index()
        if len(df) < 4:
            revenue_growth[t], op_income_growth[t], roe_values[t] = np.nan, np.nan, np.nan
            continue
        revenue_growth[t] = geometric_growth_rate(df['revenue'].tail(4))
        op_income_growth[t] = geometric_growth_rate(df['op_income'].tail(4))
        latest_equity = df['equity'].iloc[-1]
        latest_net_income = df['net_income'].iloc[-1]
        roe_values[t] = latest_net_income / latest_equity if latest_equity != 0 else np.nan

    return pd.DataFrame({
        'revenue_growth': pd.Series(revenue_growth),
        'op_income_growth': pd.Series(op_income_growth),
        'ROE': pd.Series(roe_values),
        'RSI': latest_rsi
    }).dropna()


# 포트폴리오 최적화
def markowitz_optimization(returns, cov_matrix):
    n = len(returns)
    w = cp.Variable(n)
    ret = returns.values
    cov = cov_matrix.values
    lambda_reg = 10.0

    objective = cp.Maximize((w @ (ret - RISK_FREE_RATE)) - lambda_reg * cp.quad_form(w, cov))
    constraints = [
        cp.sum(w) == 1.0,
        w >= MIN_WEIGHT,
        w <= MAX_WEIGHT
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()
    return pd.Series(w.value, index=returns.index)


# 백테스트
def quintile_ranking(df):
    ranked = pd.DataFrame(index=df.index)
    for c in df.columns:
        ranked[c] = pd.qcut(df[c].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    ranked['total_score'] = ranked.mean(axis=1)
    return ranked


def backtest(price_data, fundamental_data, rebalance_dates):
    initial_capital = 1_000_000_000
    portfolio_value = initial_capital
    portfolio_value_history, holdings = [], pd.DataFrame(index=price_data.index, columns=price_data.columns).fillna(0)

    for reb_date in rebalance_dates:
        factor_df = calculate_factors(price_data.loc[:reb_date], fundamental_data)
        if len(factor_df) < TOP_STOCKS:
            continue

        ranked = quintile_ranking(factor_df[['revenue_growth', 'op_income_growth', 'ROE', 'RSI']])
        top_candidates = ranked.sort_values('total_score', ascending=False).head(TOP_STOCKS)

        recent_returns = price_data.loc[:reb_date].pct_change().tail(60).mean()
        cov_matrix = price_data.loc[:reb_date].pct_change().tail(60).cov()

        opt_universe = top_candidates.index.intersection(recent_returns.index)
        opt_returns, opt_cov = recent_returns.loc[opt_universe], cov_matrix.loc[opt_universe, opt_universe]

        weights = markowitz_optimization(opt_returns, opt_cov)

        current_prices = price_data.loc[reb_date, opt_universe]
        allocate_amounts = weights * portfolio_value
        shares = (allocate_amounts / current_prices).fillna(0)

        holdings.loc[reb_date, :] = 0
        for s in opt_universe:
            holdings.loc[reb_date, s] = shares[s]

        next_reb_date = rebalance_dates[rebalance_dates.index(reb_date) + 1] if rebalance_dates.index(
            reb_date) + 1 < len(rebalance_dates) else price_data.index[-1]
        sub_period = price_data.loc[reb_date:next_reb_date, opt_universe]
        for d in sub_period.index:
            portfolio_value_history.append((d, (sub_period.loc[d] * shares).sum()))

    return pd.DataFrame(portfolio_value_history, columns=['date', 'portfolio_value']).set_index('date')
