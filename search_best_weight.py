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

# 기존 strategy_modules.py 내부의 함수/클래스 임포트
from strategy_modules import (
    FactorCalculator,
    calculate_2years_return,
    optimize_portfolio
)

# 기존 data_import.py 내부의 DataImporter 임포트
from data_import import DataImporter

#########################################
# 1) 팩터 가중치를 적용한 rank_stocks 함수
#########################################
def rank_stocks_with_weights(factor_df, factor_weights):
    """
    factor_df: FactorCalculator.calculate_factors() 결과 (RevenueGrowth, OpIncomeGrowth, ROE, RSI 등 열 포함)
    factor_weights: dict 형태. 예: {"RevenueGrowth":1.5, "OpIncomeGrowth":1.0, "ROE":2.0, "RSI":0.5}
        - 각 팩터별로 몇 배 가중치를 둘지 결정

    * 로직:
      1) 각 팩터별로 정규화/랭크 → score 부여
      2) score에 factor_weights를 곱해 최종 스코어를 계산
      3) 최종 스코어가 높은 순으로 종목 정렬
    """

    ranked_df = factor_df.copy()
    # 활용할 팩터 컬럼명
    factor_cols = ["RevenueGrowth", "OpIncomeGrowth", "ROE", "RSI"]

    # 안전장치: factor_weights에 모든 팩터가 들어있지 않을 경우 KeyError 방지
    for col in factor_cols:
        if col not in factor_weights:
            factor_weights[col] = 1.0  # 기본 가중치 1.0

    # (A) 각 팩터별 0~1 정규화 or 분위(Q-cut)로 점수화
    for col in factor_cols:
        # 예시) RevenueGrowth, OpIncomeGrowth -> 높은값 좋음(내림차순)
        #       ROE -> 높은값 좋음(내림차순)
        #       RSI -> 특정 구간으로 점수화(낮은 값 선호 등) 방식 등… (원하시면 커스터마이징)

        # 여기서는 간단히 "높을수록 좋음"인 팩터들을 qcut으로 1~5점 배정(ROI처럼)
        # RSI는 예시로 70 이상이면 오히려 1점(안좋음), 나머지는 5점 ~ 1점으로 나눈다고 가정
        if col != "RSI":
            # NaN이면 0으로 채우거나, 혹은 제거
            temp_series = ranked_df[col].fillna(ranked_df[col].mean())
            # 내림차순 rank
            rank_vals = temp_series.rank(method='first', ascending=False)
            # 5분위 → 5~1점
            ranked_df[col + "_score"] = 5 - pd.qcut(rank_vals, 5, labels=False, duplicates="drop")
        else:
            # RSI 70 이상은 1점, 나머지는 (낮을수록 좋다는 가정) 5점→1점
            # 여기서는 단순화: RSI < 30 → 5점 / 30~70 → 3점 / 70이상 → 1점 등으로 가능
            # (아래 예시는 qcut 활용)
            temp_rsi = ranked_df[col].fillna(ranked_df[col].mean())
            # RSI 70 이상은 일단 1점
            over_70_idx = temp_rsi >= 70
            rank_vals = temp_rsi.rank(method='first', ascending=True)  # 낮을수록 좋은 것으로 가정
            ranked_df[col + "_score"] = 5 - pd.qcut(rank_vals, 5, labels=False, duplicates="drop")
            # RSI 70 이상인 구간을 강제 1점으로 덮어씀
            ranked_df.loc[over_70_idx, col + "_score"] = 1

    # (B) 가중치 적용 → 최종 스코어
    #    예) TotalScore = sum( factor_score * factor_weight ) / (sum of weights)
    #    혹은 단순히 factor별로 곱한 뒤 모두 더한 값
    total_w = sum(factor_weights.values())
    ranked_df["TotalScore"] = 0.0
    for col in factor_cols:
        ranked_df["TotalScore"] += ranked_df[col + "_score"] * factor_weights[col]

    # 최종 스코어를 "가중치 총합"으로 나누거나 말거나는 자유.
    # 여기서는 가중합 그대로 사용.
    # ranked_df["TotalScore"] = ranked_df["TotalScore"] / total_w

    # 상위 10~20개 정도 뽑는 예시
    return ranked_df.sort_values("TotalScore", ascending=False)

#########################################
# 2) 여러 가중치 조합을 탐색(그리드 서치)하는 함수
#########################################
def search_best_factor_weights(financial_df_all,
                               date_stock_dict,
                               price_df_all,
                               data_importer,
                               factor_calc,
                               weight_candidates=None,
                               top_n=10):
    """
    financial_df_all: 모든 종목/분기의 재무데이터(이미 수집 완료)
    date_stock_dict: 날짜별(yyyymmdd) 종목 리스트
    price_df_all: yfinance에서 받아온 주가 전체 DataFrame
    data_importer: DataImporter 객체 (무위험이자율 등 가져오기 위함)
    factor_calc: FactorCalculator 객체
    weight_candidates: dict 형태의 후보군. 예) {
        'RevenueGrowth': [0.5, 1.0, 1.5],
        'OpIncomeGrowth': [0.5, 1.0, 1.5],
        'ROE': [1.0, 2.0],
        'RSI': [0.5, 1.0]
    }
    top_n: rank_stocks_with_weights 결과에서 상위 몇 종목 투자할지

    return:
        best_weights (dict) - 가장 성과 좋은 팩터 가중치
        best_performance (float) - 그때의 성과 (CAGR 등)
        all_results_df (DataFrame) - 각 가중치 조합별 성과 기록
    """
    if weight_candidates is None:
        weight_candidates = {
            'RevenueGrowth': [0.5, 1.0, 1.5],
            'OpIncomeGrowth': [0.5, 1.0, 1.5],
            'ROE': [0.5, 1.0],
            'RSI': [0.5, 1.0]
        }

    # (a) 모든 조합 만들기 (Nested loop)
    from itertools import product

    factor_keys = list(weight_candidates.keys())
    list_of_lists = [weight_candidates[k] for k in factor_keys]
    combination_list = list(product(*list_of_lists))

    # 결과 저장용
    results = []

    # (b) 단순 백테스트 파라미터 설정
    START_DATE = data_importer.start_date
    END_DATE = data_importer.end_date
    initial_money = 10_000_000

    # 너무 많은 조합을 테스트하는 경우 시간이 오래 걸리므로 유의
    for combo in combination_list:
        factor_weights = dict(zip(factor_keys, combo))
        print(f"\n[Backtest] 시도 가중치 조합: {factor_weights}")

        # 매 백테스트 마다 초기화
        current_money = initial_money
        current_date = START_DATE

        while current_date <= END_DATE:
            current_date_yyyymmdd = current_date.strftime('%Y%m%d')

            # 해당 분기의 종목 리스트
            if current_date_yyyymmdd in date_stock_dict:
                stock_codes = date_stock_dict[current_date_yyyymmdd]
                # 재무데이터 필터
                financial_df = financial_df_all[financial_df_all['Code'].isin(stock_codes)].copy()
                # 가격데이터 필터
                valid_cols = []
                for col in price_df_all.columns:
                    ticker = col.replace(".KS", "")
                    if ticker in stock_codes:
                        valid_cols.append(col)
                price_df = price_df_all[valid_cols].copy()
            else:
                # 이 날짜에 해당 종목 리스트가 없으면 break (백테스트 종료)
                break

            # 최근 4개 분기에 해당하는 YearMonth만 필터
            # 아래는 기존 get_quarterly_dates 함수를 그대로 사용:
            from simulate_backtest import get_quarterly_dates
            target_quarters = get_quarterly_dates(current_date)
            financial_df = financial_df[financial_df['YearMonth'].isin(target_quarters)]

            # 2년간 수익률 계산
            returns_df = calculate_2years_return(
                price_df,
                start_date=current_date - timedelta(days=365 * 2),
                end_date=current_date
            )

            # 팩터계산
            factor_df = factor_calc.calculate_factors(financial_df, price_df)
            common_tickers = set(factor_df['Code']).intersection(set(returns_df.columns.str.replace(".KS", "")))
            common_tickers = list(common_tickers)
            factor_df = factor_df[factor_df['Code'].isin(common_tickers)].dropna()
            returns_df.columns = returns_df.columns.str.replace(".KS", "")
            returns_df = returns_df[common_tickers]

            # (c) rank_stocks_with_weights 사용 → 상위 N 종목 선택
            ranked_df = rank_stocks_with_weights(factor_df, factor_weights)
            selected_stocks = ranked_df.head(top_n)
            selected_tickers = [code for code in selected_stocks['Code']]

            # 최적화(포트폴리오 구성)
            returns_selected = returns_df[selected_tickers].dropna(axis=0, how='any')
            tbill_rate = data_importer.get_korea_3m_tbill_rate(current_date) / 100
            opt_result, _ = optimize_portfolio(
                selected_stocks,
                returns_selected,
                risk_free_rate=tbill_rate
            )

            if opt_result['Weight'].isnull().all():
                # 최적화 해가 없으면 무위험수익률만 적용
                period_return = tbill_rate
            else:
                # 기간 수익률 계산
                from simulate_backtest import calculate_period_returns
                rebalancing_end_date = current_date + relativedelta(months=3)
                rebalancing_end_date = min(rebalancing_end_date, END_DATE)

                _, period_return = calculate_period_returns(
                    price_df,
                    selected_tickers,
                    opt_result['Weight'].values,
                    current_date,
                    rebalancing_end_date
                )

            # 자산 업데이트
            current_money = current_money * (1 + period_return)

            # 다음 분기로 이동
            current_date += relativedelta(months=3)

        # 백테스트가 끝난 시점의 final_money
        final_money = current_money
        total_years = (END_DATE - START_DATE).days / 365.25
        if total_years <= 0:
            cagr = 0
        else:
            cagr = ((final_money / initial_money) ** (1 / total_years) - 1) * 100

        # 결과 저장
        results.append({
            'RevenueGrowth_w': factor_weights['RevenueGrowth'],
            'OpIncomeGrowth_w': factor_weights['OpIncomeGrowth'],
            'ROE_w': factor_weights['ROE'],
            'RSI_w': factor_weights['RSI'],
            'CAGR': cagr,
            'FinalValue': final_money
        })

    # (d) 모든 조합 결과 정리
    all_results_df = pd.DataFrame(results)
    # CAGR가 가장 높은 조합 찾기
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

    # 1) 데이터 로드
    data_importer = DataImporter(config_path=config_path, start_date=START_DATE, rsi_period=45)
    financial_df_all, date_stock_dict = data_importer.get_all_financial_data(START_DATE, END_DATE)
    price_df_all = data_importer.get_price_data([code + ".KS" for code in financial_df_all['Code']])

    factor_calc = FactorCalculator()

    # 2) 여러 가중치 후보 설정
    weight_candidates = {
        'RevenueGrowth': [0.5, 1.0, 1.5],
        'OpIncomeGrowth': [0.5, 1.0, 1.5],
        'ROE': [0.5, 1.0],
        'RSI': [0.5, 1.0]
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
    filtered_df = all_results_df[
        (all_results_df['ROE_w'] == 1.0) & (all_results_df['RSI_w'] == 1.0)
    ]
    if not filtered_df.empty:
        plot_heatmap_for_two_factors(
            filtered_df,
            factor_x='RevenueGrowth_w',
            factor_y='OpIncomeGrowth_w',
            value_col='CAGR'
        )
        plot_heatmap_for_two_factors(
            filtered_df,
            factor_x='RevenueGrowth',
            factor_y='ROE_w',
            value_col='CAGR'
        )
        plot_heatmap_for_two_factors(
            filtered_df,
            factor_x='RevenueGrowth',
            factor_y='RSI_w',
            value_col='CAGR'
        )
        plot_heatmap_for_two_factors(
            filtered_df,
            factor_x='OpIncomeGrowth_w',
            factor_y='ROE_w',
            value_col='CAGR'
        )
        plot_heatmap_for_two_factors(
            filtered_df,
            factor_x='OpIncomeGrowth_w',
            factor_y='RSI_w',
            value_col='CAGR'
        )
        plot_heatmap_for_two_factors(
            filtered_df,
            factor_x='ROE_w',
            factor_y='RSI_w',
            value_col='CAGR'
        )
    else:
        print("해당 조건(ROE=1.0, RSI=1.0)에 해당하는 결과가 없어 heatmap을 표시하지 않습니다.")