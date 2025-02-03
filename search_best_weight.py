"""
search_best_weight.py

기존 FactorCalculator의 rank_stocks를 변형하여
각 팩터별 가중치 조합을 탐색(Grid Search)해보고,
해당 가중치로 뽑은 종목들로 포트폴리오를 구성하는 백테스트를 수행해
가장 좋은 성과(예: CAGR, Sharpe Ratio 등)를 내는 가중치 조합을 찾는 예시 코드.

최종적으로는 성과 요약과 함께, 가중치 조합별 결과를
Heatmap 혹은 표로 시각화하는 과정을 포함합니다.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
import json

# 기존 strategy_modules.py 내부의 함수/클래스 임포트
from strategy_modules import (
    FactorCalculator,
    calculate_2years_return,
    optimize_portfolio
)

# 기존 data_import.py 내부의 DataImporter 임포트
from data_import import DataImporter

#FutureWarning 제거
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#UserWarning 제거
warnings.simplefilter(action='ignore', category=UserWarning)

#########################################
# 1) 팩터 가중치를 적용한 rank_stocks 함수
#########################################
def rank_stocks_with_weights(factor_df, factor_weights):
    """
    수정된 rank_stocks_with_weights 함수
    - RevenueGrowth와 OpIncomeGrowth: min-max 정규화(normalize_series 사용) 후 내림차순 순위 산출
    - ROE: 정규화 없이 내림차순 순위 산출
    - RSI: 내림차순 순위 산출하되, RSI 값이 70 이상이면 강제로 1점 부여
    - 동적 가중치는 factor_weights 인자로 받아 각 팩터 점수에 곱해 최종 TotalScore를 계산함
    """
    from strategy_modules import normalize_series
    # factor_df를 복사하여 작업
    ranked_df = factor_df.copy()
    factor_cols = ["RevenueGrowth", "OpIncomeGrowth", "ROE", "RSI"]

    # 모든 팩터에 대해 동적 가중치가 없으면 기본값 1.0 할당
    for col in factor_cols:
        if col not in factor_weights:
            print("Warning: No weight assigned for", col)
            print("Default weight 1.0 is assigned.")
            factor_weights[col] = 1.0

    # 각 팩터별 점수 산출
    for col in factor_cols:
        if col in ["RevenueGrowth", "OpIncomeGrowth"]:
            # 음수값 처리 등으로 normalize_series를 적용하여 [0,1] 범위로 변환
            ranked_df[col + "_normalized"] = normalize_series(ranked_df[col])
            # 내림차순 순위: 값이 클수록 더 높은 순위를 부여
            rank_vals = ranked_df[col + "_normalized"].rank(method='first', ascending=False)
            # 5분위(qcut)를 사용하여 5~1 점수 산출 (5점: 상위 20%, 1점: 하위 20%)
            ranked_df[col + "_score"] = 5 - pd.qcut(rank_vals, 5, labels=False, duplicates="drop")

        elif col == "ROE":
            # 모든 ROE값이 NaN이면 기본 1점 할당
            if ranked_df[col].isna().all():
                ranked_df[col + "_score"] = 1
            else:
                # 정규화 없이 내림차순 순위 산출
                rank_vals = ranked_df[col].rank(method='first', ascending=False)
                ranked_df[col + "_score"] = 5 - pd.qcut(rank_vals, 5, labels=False, duplicates="drop")

        elif col == "RSI":
            # 모든 RSI값이 NaN이면 기본 1점 할당
            if ranked_df[col].isna().all():
                ranked_df[col + "_score"] = 1
            else:
                # 내림차순 순위 산출: 높은 값이 좋은 것으로 가정
                rank_vals = ranked_df[col].rank(method='first', ascending=False)
                ranked_df[col + "_score"] = 5 - pd.qcut(rank_vals, 5, labels=False, duplicates="drop")
                # 단, RSI 값이 70 이상이면 과매수로 판단하여 강제로 1점 할당
                ranked_df.loc[ranked_df[col] >= 70, col + "_score"] = 1

    # 동적 가중치를 적용하여 TotalScore 계산 (가중합 방식)
    ranked_df["TotalScore"] = (
            ranked_df["RevenueGrowth_score"] * factor_weights["RevenueGrowth"] +
            ranked_df["OpIncomeGrowth_score"] * factor_weights["OpIncomeGrowth"] +
            ranked_df["ROE_score"] * factor_weights["ROE"] +
            ranked_df["RSI_score"] * factor_weights["RSI"]
    )

    # TotalScore 기준 내림차순 정렬 후 반환
    return ranked_df.sort_values("TotalScore", ascending=False)

#########################################
# 추가 - CAGR 계산 함수
#########################################
def calculate_cagr(initial_money, final_money, start_date, end_date):
    total_years = max((end_date - start_date).days / 365.25, 0.0001)  # 0으로 나누기 방지
    if final_money <= 0:  # 손실이 -100%를 넘는 경우
        return -100.0
    return (((final_money / initial_money) ** (1 / total_years)) - 1) * 100

#########################################
# 2) 여러 가중치 조합을 탐색(그리드 서치)하는 함수
#########################################
def run_single_combination(combo_args):
    """
    combo_args: (factor_weights_dict, 기타 필요한 인자들...) 튜플 또는 dict

    이 함수가 실제로 해당 팩터 가중치 조합으로 백테스트를 수행한 뒤,
    결과(CAGR, FinalValue 등)를 리턴한다고 가정
    """
    (factor_weights, financial_df_all, date_stock_dict, price_df_all,
     data_importer, factor_calc, top_n) = combo_args

    # ========== 실제 백테스트 로직 시작 (기존 search_best_factor_weights의 루프 부분) ==========

    # 매 백테스트 마다 초기화
    START_DATE = data_importer.start_date
    END_DATE = data_importer.end_date
    initial_money = 10_000_000
    current_money = initial_money
    current_date = START_DATE

    # 날짜 순서 보정
    dates = sorted([d for d in date_stock_dict.keys() if datetime.strptime(d, '%Y%m%d') >= START_DATE])

    for current_date_yyyymmdd in dates:
        current_date = datetime.strptime(current_date_yyyymmdd, '%Y%m%d')
        if current_date > END_DATE:
            break

        stock_codes = date_stock_dict[current_date_yyyymmdd]
        financial_df = financial_df_all[financial_df_all['Code'].isin(stock_codes)].copy()
        valid_cols = []
        for col in price_df_all.columns:
            ticker = col.replace(".KQ", "")
            if ticker in stock_codes:
                valid_cols.append(col)
        price_df = price_df_all[valid_cols].copy()

        # 2년간 수익률 계산
        from simulate_backtest import get_quarterly_dates, calculate_period_returns
        from strategy_modules import calculate_2years_return, optimize_portfolio

        target_quarters = get_quarterly_dates(current_date)
        financial_df = financial_df[financial_df['YearMonth'].isin(target_quarters)]
        returns_df = calculate_2years_return(
            price_df,
            start_date=current_date - timedelta(days=365 * 2),
            end_date=current_date
        )

        # 팩터 계산
        factor_df = factor_calc.calculate_factors(financial_df, price_df)
        common_tickers = set(factor_df['Code']).intersection(
            set(returns_df.columns.str.replace(".KQ", ""))
        )
        factor_df = factor_df[factor_df['Code'].isin(common_tickers)].dropna()
        returns_df.columns = returns_df.columns.str.replace(".KQ", "")
        returns_df = returns_df[list(common_tickers)]

        # (c) rank_stocks_with_weights 사용 → 상위 N 종목
        ranked_df = rank_stocks_with_weights(factor_df, factor_weights)
        selected_stocks = ranked_df.head(top_n)
        selected_tickers = selected_stocks['Code'].tolist()

        # 최적화(포트폴리오 구성)
        returns_selected = returns_df[selected_tickers].dropna(axis=0, how='any')
        tbill_rate = data_importer.get_korea_3m_tbill_rate(current_date) / 100
        opt_result, _ = optimize_portfolio(selected_stocks, returns_selected, risk_free_rate=tbill_rate)

        # 수익률 계산
        if opt_result['Weight'].isnull().all():
            # 최적화 해가 없으면 무위험수익률만 적용
            period_return = (data_importer.get_korea_3m_tbill_rate(current_date) / 100)
        else:
            rebalancing_end_date = current_date + relativedelta(months=3)
            rebalancing_end_date = min(rebalancing_end_date, END_DATE)
            _, period_return = calculate_period_returns(
                price_df,
                selected_tickers,
                opt_result['Weight'].values,
                current_date,
                rebalancing_end_date
            )
        # 수익률 에러 발생 처리
        if abs(period_return) > 1.0:
            print(f"Error: Abnormal return detected at {current_date}, period return: {period_return}\n it set to 0.0")
            period_return = 0.0

        # 자산 업데이트
        current_money *= (1 + period_return)

        # 다음 분기로 이동
        current_date += relativedelta(months=3)

    # 백테스트 결과
    final_money = current_money
    total_years = (END_DATE - START_DATE).days / 365.25
    cagr = calculate_cagr(initial_money, final_money, START_DATE, END_DATE)

    # 리턴할 딕셔너리
    return {
        'RevenueGrowth_w': factor_weights['RevenueGrowth'],
        'OpIncomeGrowth_w': factor_weights['OpIncomeGrowth'],
        'ROE_w': factor_weights['ROE'],
        'RSI_w': factor_weights['RSI'],
        'CAGR': cagr,
        'FinalValue': final_money
    }
def search_best_factor_weights(financial_df_all,
                               date_stock_dict,
                               price_df_all,
                               data_importer,
                               factor_calc,
                               weight_candidates=None,
                               top_n=10):
    from itertools import product
    import multiprocessing  # 여기 추가

    factor_keys = list(weight_candidates.keys())
    list_of_lists = [weight_candidates[k] for k in factor_keys]
    combination_list = list(product(*list_of_lists))

    # (1) 풀에서 돌릴 준비: 각 조합에 필요한 인자 패키징
    task_args_list = []
    for combo in combination_list:
        factor_weights = dict(zip(factor_keys, combo))
        # run_single_combination에 들어가는 인자들
        args = (
            factor_weights,
            financial_df_all,
            date_stock_dict,
            price_df_all,
            data_importer,
            factor_calc,
            top_n
        )
        task_args_list.append(args)

    # (2) multiprocessing Pool 사용
    cpu_count = max(multiprocessing.cpu_count()-1, 1)  # 또는 원하는 만큼
    with multiprocessing.Pool(processes=cpu_count) as pool:
        results = pool.map(run_single_combination, task_args_list)
    # 이때 results에는 각 조합별로 return된 dict가 순서대로 들어감

    # (3) 결과를 DataFrame으로 정리
    all_results_df = pd.DataFrame(results)

    # (4) 최고 CAGR 찾기
    best_row = all_results_df.loc[all_results_df['CAGR'].idxmax()]
    best_weights = {
        'RevenueGrowth': best_row['RevenueGrowth_w'],
        'OpIncomeGrowth': best_row['OpIncomeGrowth_w'],
        'ROE': best_row['ROE_w'],
        'RSI': best_row['RSI_w']
    }
    best_performance = best_row['CAGR']

    return best_weights, best_performance, all_results_df


#########################################
# 3) Heatmap 등 시각화 예시 함수
#########################################
def plot_heatmap_for_two_factors (results_df,
                                 factor_x='RevenueGrowth_w',
                                 factor_y='OpIncomeGrowth_w',
                                 value_col='CAGR'):
    """
    2개의 팩터 가중치만 고정하고 나머지는 필터 or 단일값으로 맞출 때,
    그 결과를 Heatmap으로 표현하는 예시.
    (4차원 전부를 Heatmap으로 표현하긴 어려우므로,
     2개 팩터만 축으로 잡는 예시를 보여줌)
    """
    pivot_df = results_df.pivot_table(
        index=factor_y,
        columns=factor_x,
        values=value_col,
        aggfunc='mean'
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_df, annot=True, cmap='RdYlBu', fmt=".2f")
    plt.title(f"Heatmap: {value_col} by {factor_x} vs {factor_y}")
    plt.xlabel(factor_x)
    plt.ylabel(factor_y)
    plt.tight_layout()
    plt.show(block=True)


#########################################
# 4) 메인 구동부
#########################################
if __name__ == "__main__":
    from simulate_backtest import get_quarterly_dates, calculate_period_returns

    config_path = 'config.json'
    START_DATE = datetime(2018, 5, 15)
    END_DATE = datetime.now()

    # 1) 데이터 로드, 저장되어 있다면 로드해서 불러오기, 저장되어 있지 않다면 다시 수집 후 저장
    data_importer = DataImporter(config_path=config_path, start_date=START_DATE, rsi_period=45)
    financial_df_path = 'financial_df_all.csv'
    price_df_path = 'price_df_all.csv'
    date_stock_dict_path = 'date_stock_dict.json'

    if os.path.exists(financial_df_path) and os.path.exists(price_df_path) and os.path.exists(date_stock_dict_path):
        financial_df_all = pd.read_csv(financial_df_path, dtype={'Code': str, 'YearMonth': str})
        price_df_all = pd.read_csv(price_df_path, parse_dates=['Date'])
        price_df_all.set_index('Date', inplace=True)
        with open(date_stock_dict_path, 'r') as f:
            date_stock_dict = json.load(f)
        print("데이터 로드 완료")
    else:
        financial_df_all, date_stock_dict = data_importer.get_all_financial_data(START_DATE, END_DATE)
        price_df_all = data_importer.get_price_data([code + ".KQ" for code in financial_df_all['Code']])

        financial_df_all.to_csv(financial_df_path, index=True)
        json.dump(date_stock_dict, open(date_stock_dict_path, 'w'))
        price_df_all.to_csv(price_df_path, index=True)
        print("데이터 수집 및 저장 완료")

    factor_calc = FactorCalculator()

    # 2) 여러 가중치 후보 설정
    weight_candidates = {
        'RevenueGrowth': [1],
        'OpIncomeGrowth': [1],
        'ROE': list(np.arange(0.3, 2.2, 0.2)),
        'RSI': list(np.arange(0.3, 2.2, 0.2))
    }

    # 3) 탐색
    best_weights, best_perf, all_results_df = search_best_factor_weights(
        financial_df_all=financial_df_all,
        date_stock_dict=date_stock_dict,
        price_df_all=price_df_all,
        data_importer=data_importer,
        factor_calc=factor_calc,
        weight_candidates=weight_candidates,
        top_n=11
    )

    print("\n===========================")
    print("최적 가중치 조합:", best_weights)
    print(f"해당 조합의 CAGR: {best_perf:.2f}%")

    # 4) Heatmap 예시: (RevenueGrowth vs OpIncomeGrowth)에 따른 CAGR
    #    나머지 ROE, RSI는 한 가지 값만 필터하거나 평균
    #    (4차원 전부를 한 장의 Heatmap으로 표현은 불가하므로 예시)
    #    - 아래는 ROE=1.0, RSI=1.0인 경우만 필터하여 Heatmap
    filtered_df = all_results_df
    if not filtered_df.empty:
        plot_heatmap_for_two_factors(
            filtered_df,
            factor_x='ROE_w',
            factor_y='RSI_w',
            value_col='CAGR'
        )
    else:
        print("해당 조건(ROE=1.0, RSI=1.0)에 해당하는 결과가 없어 heatmap을 표시하지 않습니다.")