# config.py
# 설정 정보: 기간, 리밸런싱 주기, 최소/최대 비중, 종목 수 등.

START_DATE = "2015-01-01"
END_DATE = "2020-12-31"
REBALANCE_FREQ = "Q"  # 분기마다 리밸런싱
TOP_STOCKS = 11
MIN_WEIGHT = 0.01
MAX_WEIGHT = 0.20
RISK_FREE_RATE = 0.015  # 예시적인 무위험 수익률

# RSI기간
RSI_PERIOD = 30

# 마켓 유니버스 파일 위치
KOSPI200_LIST_PATH = "./data/kospi200_list.csv"
