# main.py

from data_import import DataImporter
from strategy_modules import FactorCalculator, optimize_portfolio
import pandas as pd
import json
import os

##1. config.json 로드
config_path = 'config.json'
if not os.path.exists(config_path):
    raise FileNotFoundError(f"{config_path} not found.")

# 2. Data Importer 인스턴스 생성
importer = DataImporter(config_path=config_path)

# 3. KOSPI200 종목 리스트에 대한 매출, 영업이익, ROE 데이터 수집
financial_df = importer.get_all_financial_data()
# Year과 Report 컬럼 추가
financial_df['Year'] = financial_df['YearMonth'].astype(str).str[:4].astype(int)
financial_df['Report'] = financial_df['YearMonth'].astype(str).str[4:].astype(int)
print("Available columns:", financial_df.columns)

# 4. 주가데이터 수집
print("주가데이터 수집 중...")
tickers = [code + ".KS" for code in financial_df['Code']]
price_df = importer.get_price_data(tickers)
## 6. 팩터 계산
factor_calc = FactorCalculator()
factor_df = factor_calc.calculate_factors(financial_df, price_df)
print("팩터 계산 완료.")

# 7. 팩터 랭킹 및 상위종목 선별
selected_stocks = factor_calc.rank_stocks(factor_df)
print(f"선정된 종목 수: {len(selected_stocks)}")
##
if selected_stocks.empty:
    print("선정된 종목이 없습니다. 전략을 재검토해주세요.")

# 8. 포트폴리오 최적화
print("포트폴리오 최적화 중...")
# 수익률 계산
returns_df = price_df.pct_change().dropna()

# 선택된 종목의 수익률 필터링
selected_tickers = [code  for code in selected_stocks['Code']]
# returns_df의 열 이름과 selected_tickers의 일치 여부 확인 및 수정
returns_df.columns = returns_df.columns.str.replace(".KS", "", regex=False)  # ".KS" 제거
# returns_selected 필터링
returns_selected = returns_df[selected_tickers].dropna()


## 최적화 수행
opt_result = optimize_portfolio(selected_stocks, returns_selected, risk_free_rate=0.001)

if opt_result is not None:
    print("최적화된 포트폴리오:")
    print(opt_result)
else:
    print("포트폴리오 최적화 실패.")

##

