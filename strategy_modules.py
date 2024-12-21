# strategy_modules.py

import pandas as pd
import numpy as np
import cvxpy as cp


def calculate_growth_rate(series):
    # 4분기 복리성장률 = (마지막값 / 첫값)^(1/4)-1
    if len(series) < 4:
        return np.nan
    return (series.iloc[-1] / series.iloc[0]) ** (1 / 4) - 1


def calculate_rsi(price_series, period=30):
    # RSI 계산
    delta = price_series.diff()
    up = delta.where(delta > 0, 0)
    down = -delta.where(delta < 0, 0)
    roll_up = up.rolling(window=period).mean()
    roll_down = down.rolling(window=period).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


class FactorCalculator:
    def __init__(self):
        pass

    def calculate_factors(self, financial_df, price_df):
        """
        financial_df: 재무데이터 DataFrame
        price_df: 주가 데이터 DataFrame
        """
        # 종목별로 재무데이터 그룹화
        grouped = financial_df.groupby('Code')

        factor_dict = {
            'Code': [],
            'RevenueGrowth': [],
            'OpIncomeGrowth': [],
            'ROE': [],
            'RSI': []
        }

        for code, group in grouped:
            group = group.sort_values(['YearMonth', 'Report'])

            # 매출액과 영업이익의 성장률 계산
            rev_growth = calculate_growth_rate(group['매출액'])
            op_growth = calculate_growth_rate(group['영업이익'])

            # ROE는 가장 최근 값 사용
            roe = group['ROE'].iloc[-1] if not group['ROE'].isnull().all() else np.nan

            # RSI 계산
            ticker = code + ".KS"
            if ticker in price_df.columns:
                rsi_series = calculate_rsi(price_df[ticker])
                rsi_val = rsi_series.iloc[-1] if not rsi_series.dropna().empty else np.nan
            else:
                rsi_val = np.nan

            factor_dict['Code'].append(code)
            factor_dict['RevenueGrowth'].append(rev_growth)
            factor_dict['OpIncomeGrowth'].append(op_growth)
            factor_dict['ROE'].append(roe)
            factor_dict['RSI'].append(rsi_val)

        factor_df = pd.DataFrame(factor_dict)
        return factor_df

    def rank_stocks(self, factor_df):
        """
        각 Factor별 Quintile 나눔 후 1~5점 할당 (5점: 상위 20%, 1점: 하위 20%)
        """
        ranked_df = factor_df.copy()

        for col in ['RevenueGrowth', 'OpIncomeGrowth', 'ROE', 'RSI']:
            ranked_df[col + '_score'] = ranked_df[col].rank(method='first', ascending=False)
            ranked_df[col + '_score'] = pd.qcut(ranked_df[col + '_score'], 5, labels=False) + 1  # 1~5
            ranked_df[col + '_score'] = 6 - ranked_df[col + '_score']  # 상위에 높은 점수

        # 종합 점수 계산 (평균)
        ranked_df['TotalScore'] = ranked_df[
            ['RevenueGrowth_score', 'OpIncomeGrowth_score', 'ROE_score', 'RSI_score']].mean(axis=1)

        # 상위 20% 종목 필터
        cutoff = ranked_df['TotalScore'].quantile(0.8)
        selected = ranked_df[ranked_df['TotalScore'] >= cutoff].reset_index(drop=True)
        return selected


def optimize_portfolio(selected_stocks, returns_df):
    """
    Markowitz 최적화
    selected_stocks: 선택된 종목 DataFrame
    returns_df: 선택된 종목의 일별 수익률 DataFrame
    """
    # 기대 수익률 및 공분산 계산
    mu = returns_df.mean()
    Sigma = returns_df.cov()
    codes = selected_stocks['Code'].tolist()

    mu_selected = mu[codes].values
    Sigma_selected = Sigma.loc[codes, codes].values

    # 변수 정의
    w = cp.Variable(len(codes))

    # 목적 함수: 최대 Sharpe Ratio (평균 수익률 / 포트폴리오 변동성)
    # 이를 직접 최적화하기는 어렵기 때문에, 다음과 같이 변환
    # Maximize mu^T w - lambda * w^T Sigma w
    lambda_ = 1.0
    objective = cp.Maximize(mu_selected @ w - lambda_ * cp.quad_form(w, Sigma_selected))

    # 제약조건
    constraints = [
        cp.sum(w) == 1,
        w >= 0.01,
        w <= 0.20
    ]

    # 문제 정의 및 해결
    prob = cp.Problem(objective, constraints)
    prob.solve()

    if w.value is None:
        print("Optimization failed.")
        return None

    weights = w.value
    result = pd.DataFrame({'Code': codes, 'Weight': weights})
    return result
