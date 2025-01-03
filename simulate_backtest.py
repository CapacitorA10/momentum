# simulate_backtest.py

from data_import import DataImporter
from strategy_modules import FactorCalculator, optimize_portfolio, calculate_2years_return
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

# input 날짜로부터 직전 4개 분기를 출력 (4분기 복리성장률 계산을 위함)
def get_quarterly_dates(current_date):
    # 분기 마감 후 45일 후의 데이터임을 가정하여, 1개월치 데이터 차감 필요
    current_date = current_date - timedelta(days=15) - relativedelta(months=1)
    output_dates = []
    for i in range(4):
        current_year = current_date.year
        current_month = current_date.month
        output_dates.append(f"{current_year}{str(current_month).zfill(2)}")
        # Current_date 를 3개월 전으로 변경, days 사용 아닌 3개월 차감 필요
        current_date = current_date - relativedelta(months=3)

    return output_dates

# 기간 손익 계산
def calculate_period_returns(price_df, tickers, weights, start_date, end_date):
    # 기간에 맞게 주가 필터링, 'KS' 제거
    price_df.columns = price_df.columns.str.replace(".KS", "", regex=False)
    price_df = price_df.loc[start_date:end_date, tickers]
    daily_returns = price_df.pct_change().fillna(0)

    # 일별 가중평균 수익률
    portfolio_daily_returns = daily_returns.dot(weights)

    # 누적 수익률 계산
    cumulative_returns = (1 + portfolio_daily_returns).cumprod()

    return cumulative_returns, cumulative_returns[-1] - 1 # 누적 수익률, 기간 수익률

## 백테스트 초기 셋업
config_path = 'config.json'
data_importer = DataImporter(config_path=config_path)
factor_calc = FactorCalculator()

START_DATE = datetime(2018, 5, 15)
END_DATE = datetime(2024, 12, 30)

financial_df_all = data_importer.get_all_financial_data()
price_df_all = data_importer.get_price_data([code + ".KS" for code in financial_df_all['Code']])

## 백테스팅 진행
initial_money = 10000000 # 1천만원
current_money = initial_money
money_history = []
daily_value_history = pd.DataFrame()
date_history = []
current_date = START_DATE

while current_date <= END_DATE:
    print(f"Processing on {current_date.strftime('%Y-%m-%d')}")

    # 1. 재무데이터 및 주가 데이터 수집
    financial_df = financial_df_all.copy()
    price_df = price_df_all.copy()

    # 2. current_date 에 맞게 데이터 필터링
    target_quarters = get_quarterly_dates(current_date)
    financial_df = financial_df[financial_df['YearMonth'].isin(target_quarters)]
    returns_df = calculate_2years_return(price_df, start_date=current_date-timedelta(days=365*2), end_date=current_date)

    # 3. 팩터 계산 및 데이터 정렬
    factor_df = factor_calc.calculate_factors(financial_df, price_df)
    common_tickers = set(factor_df['Code']).intersection(set(returns_df.columns.str.replace(".KS", "", regex=False)))
    common_tickers = list(common_tickers)
    factor_df = factor_df[factor_df['Code'].isin(common_tickers)].dropna()
    returns_df.columns = returns_df.columns.str.replace(".KS", "", regex=False)
    returns_df = returns_df[common_tickers]

    # 4. 팩터 랭킹 및 상위종목 선별 후 포트폴리오 최적화
    selected_stocks = factor_calc.rank_stocks(factor_df)
    selected_tickers = [code for code in selected_stocks['Code']]
    returns_df.columns = returns_df.columns.str.replace(".KS", "", regex=False)
    returns_selected = returns_df[selected_tickers].dropna()
    opt_result, sharpe_ratio = optimize_portfolio(selected_stocks, returns_selected, risk_free_rate=0.001)

    # 5. 포트폴리오 비중에 따른 수익 계산
    weights = opt_result['Weight'].values

    # 기간 수익률 계산
    rebalancing_end_date = current_date + relativedelta(months=3)
    rebalancing_end_date = min(rebalancing_end_date, END_DATE)  # 종료일을 초과하지 않도록
    daily_returns, period_return = calculate_period_returns(price_df, selected_tickers, weights, current_date,
                                                            rebalancing_end_date)

    # 6. 포트폴리오 잔고 업데이트
    current_money_before = current_money
    current_money *= (1 + period_return)
    money_history.append(current_money)
    date_history.append(current_date)

    daily_asset_values = current_money_before * daily_returns
    daily_asset_values = daily_asset_values.to_frame(name="PortfolioValue")
    daily_value_history = pd.concat([daily_value_history, daily_asset_values])

    # 다음 분기로 이동
    current_date += relativedelta(months=3)

## 그래프 그리기
# BM(코스피) 데이터 가져오기
kospi_data = data_importer.fetch_stock_price("^KS200", daily_value_history.index.min(), daily_value_history.index.max())

plt.plot(daily_value_history.index, daily_value_history["PortfolioValue"], label="Portfolio Value", color="blue")
plt.plot(kospi_data.index, kospi_data["Cumulative Return"] * initial_money, label="KOSPI (Cumulative Return)", color="green")

plt.title("Portfolio Value vs KOSPI Index", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Portfolio Value (KRW)", fontsize=12)
plt.grid(True)
plt.legend()
plt.show(block=True)


##

plt.close()
# CAGR, MDD, Sharpe Ratio 계산 필요