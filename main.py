# main.py

from data_import import DataImporter
from strategy_modules import FactorCalculator, optimize_portfolio, calculate_2years_return
import pandas as pd
import json
import os
from datetime import datetime

##1. config.json 로드
config_path = 'config.json'
if not os.path.exists(config_path):
    raise FileNotFoundError(f"{config_path} not found.")

# 2. Data Importer 인스턴스 생성
importer = DataImporter(config_path=config_path)

# 3. KOSPI200 종목 리스트에 대한 매출, 영업이익, ROE 데이터 수집
financial_df = importer.get_all_financial_data()
print("Available columns:", financial_df.columns)

# 4. 주가데이터 수집
print("주가데이터 수집 중...")
tickers = [code + ".KS" for code in financial_df['Code']]
price_df = importer.get_price_data(tickers)

# 5. 2년치 기대수익률 계산, 현재 날짜로부터 2년 전까지
start_date = datetime.now().replace(year=datetime.now().year - 2)
end_date = datetime.now()
returns_df = calculate_2years_return(price_df, start_date=start_date, end_date=end_date)

## 6. 팩터 계산
factor_calc = FactorCalculator()
factor_df = factor_calc.calculate_factors(financial_df, price_df)
print("팩터 계산 완료.")
# 교집합만 남기기
common_tickers = set(factor_df['Code']).intersection(set(returns_df.columns.str.replace(".KS", "", regex=False)))
common_tickers = list(common_tickers)
factor_df = factor_df[factor_df['Code'].isin(common_tickers)]
returns_df.columns = returns_df.columns.str.replace(".KS", "", regex=False)
returns_df = returns_df[common_tickers]

# 7. 팩터 랭킹 및 상위종목 선별
selected_stocks = factor_calc.rank_stocks(factor_df)
print(f"선정된 종목 수: {len(selected_stocks)}")

# 8. 포트폴리오 최적화
# 선택된 종목의 수익률 필터링
selected_tickers = [code for code in selected_stocks['Code']]
# returns_df의 열 이름과 selected_tickers의 일치 여부 확인 및 수정
returns_df.columns = returns_df.columns.str.replace(".KS", "", regex=False)  # ".KS" 제거
returns_selected = returns_df[selected_tickers].dropna()


## 최적화 수행
opt_result = optimize_portfolio(selected_stocks, returns_selected, risk_free_rate=0.001)

if opt_result is not None:
    print("최적화된 포트폴리오:")
    print(opt_result)
else:
    print("포트폴리오 최적화 실패.")

##

