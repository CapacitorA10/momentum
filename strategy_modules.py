# strategy_modules.py

import pandas as pd
import numpy as np
import cvxpy as cp

def calculate_2years_return(price_df, start_date, end_date):
    returns_df = price_df.pct_change()
    # 월간 수익률로 보정
    returns_df = (1 + returns_df).resample('ME').prod() - 1
    returns_df = returns_df[(returns_df.index >= start_date) & (returns_df.index <= end_date)]
    returns_df = returns_df.dropna(axis=1, how='any')
    return returns_df

def calculate_growth_rate(series):
    # 4분기 복리성장률 = 뒤에서 4번째 값 / 첫번째 값의 1/4 제곱 - 1
    if len(series) < 4:
        return np.nan
    first = series.iloc[-4]
    last = series.iloc[-1]
    if first == 0:
        return np.nan
    ratio = last / first
    if ratio <= 0:
        return -1
    return ratio ** (1 / 4) - 1


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
            if col in ['RevenueGrowth', 'OpIncomeGrowth']:
                # 음수 값 별도처리
                valid = ranked_df[col] >= 0
                ranked_df.loc[valid, col + '_rank'] = ranked_df.loc[valid, col].rank(method='first', ascending=False)
                ranked_df.loc[~valid, col + '_rank'] = 0 # 음수정장률은 최하위
                # quintile 계산
                ranked_df[col + '_score'] = pd.qcut(ranked_df.loc[valid, col + '_rank'], 5, labels=False) + 1  # 1~5
                ranked_df[col + '_score'] = 6 - ranked_df[col + '_score']  # 상위에 높은 점수
                # 음수값은 무조건 1점
                ranked_df.loc[~valid, col + '_score'] = 1
            else: # ROE, RSI
                ranked_df[col + '_score'] = ranked_df[col].rank(method='first', ascending=False)
                ranked_df[col + '_score'] = pd.qcut(ranked_df[col + '_score'], 5, labels=False) + 1  # 1~5
                ranked_df[col + '_score'] = 6 - ranked_df[col + '_score']  # 상위에 높은 점수
                # RSI 70 이상은 1점(과매수 구간)
                ranked_df.loc[ranked_df[col] >= 70, col + '_score'] = 1

        # 종합 점수 계산 (평균)
        ranked_df['TotalScore'] = ranked_df[
            ['RevenueGrowth_score', 'OpIncomeGrowth_score', 'ROE_score', 'RSI_score']].mean(axis=1)

        # 상위 11개 종목 추출
        ranked_df = ranked_df.sort_values('TotalScore', ascending=False).head(11)
        return ranked_df



def optimize_portfolio(selected_stocks, returns_df, approach='bisection',
                       risk_free_rate=0.03, min_weight=0.01, max_weight=0.20,
                       max_iter=50, lam=1.0):
    """
    selected_stocks: 선정된 종목 DataFrame (e.g., ['Code'] 열 포함)
    returns_df: 선정된 종목의 일별 수익률 DataFrame (columns = 종목코드)
    approach: 'bisection' 또는 'mean_variance'
        - 'bisection': Sharpe Ratio 최대화(이진 탐색)
        - 'mean_variance': 평균-분산(Markowitz) 접근
    risk_free_rate: 무위험수익률
    min_weight: 최소 비중
    max_weight: 최대 비중
    max_iter: bisection 반복 횟수
    lam: mean-variance 방식에서 위험회피계수(lambda)

    return:
        최적화된 포트폴리 DataFrame (각 종목 코드와 최적화된 Weight)
        + Sharpe Ratio (또는 목적함수값)
    """
    # 1) 기대수익률(mu) 및 공분산(Sigma) 계산
    mu = returns_df.mean()   # 기대 수익률(평균)
    Sigma = returns_df.cov() # 공분산 행렬

    # 2) selected_stocks에서 종목 코드만 추출
    codes = selected_stocks['Code'].tolist()

    # 3) mu, Sigma도 선택된 종목에 맞춰 필터링
    mu_selected = mu[codes].values
    Sigma_selected = Sigma.loc[codes, codes].values

    # 4) approach 분기
    if approach == 'bisection':
        # -- Sharpe Ratio 최대화 (이진 탐색) --
        best_alpha, best_w = _maximize_sharpe_ratio_bisection(
            mu_selected, Sigma_selected,
            risk_free_rate=risk_free_rate,
            min_weight=min_weight,
            max_weight=max_weight,
            max_iter=max_iter
        )
        # 결과 DataFrame 정리
        result_df = pd.DataFrame({'Code': codes, 'Weight': best_w})
        return result_df, best_alpha

    elif approach == 'mean_variance':
        # -- Mean-Variance (Markowitz) 접근 --
        w_opt = _optimize_mean_variance(
            mu_selected, Sigma_selected,
            risk_free_rate=risk_free_rate,
            lam=lam,
            min_weight=min_weight,
            max_weight=max_weight
        )
        # 결과 DataFrame
        result_df = pd.DataFrame({'Code': codes, 'Weight': w_opt})

        # (참고) Mean-Variance 해에서의 Sharpe Ratio도 간단 추정 가능
        sr = _compute_sharpe_ratio(w_opt, mu_selected, Sigma_selected, risk_free_rate)
        return result_df, sr

    else:
        raise ValueError("approach는 'bisection' 또는 'mean_variance' 중 하나여야 합니다.")


# -----------------------
# 아래는 내부적으로 사용하는 보조함수들
# -----------------------

def _maximize_sharpe_ratio_bisection(mu, Sigma, risk_free_rate=0.03,
                                     min_weight=0.01, max_weight=0.20, max_iter=50):
    """
    Sharpe Ratio = (mu - rf)^T w / sqrt(w^T Sigma w)
    를 이진 탐색(bisection)으로 최대화
    """
    alpha_low = 0.0
    alpha_high = 10.0  # Sharpe Ratio 최대 범위를 임의로 넉넉하게 잡음
    best_alpha = 0.0
    best_w = None

    for _ in range(max_iter):
        alpha_mid = 0.5 * (alpha_low + alpha_high)
        feasible, w_candidate = _check_feasibility_sharpe(
            alpha_mid, mu, Sigma, risk_free_rate,
            min_weight, max_weight
        )

        if feasible:
            # alpha_mid 달성 가능 -> 더 큰 Sharpe ratio 시도
            best_alpha = alpha_mid
            best_w = w_candidate
            alpha_low = alpha_mid
        else:
            # alpha_mid 달성 불가 -> Sharpe ratio 하향 조정
            alpha_high = alpha_mid

    return best_alpha, best_w


def _check_feasibility_sharpe(alpha, mu, Sigma, risk_free_rate,
                              min_weight, max_weight):
    """
    alpha 이상 Sharpe Ratio 달성 가능한 w 존재 여부를 확인.
    - (mu - rf)^T w >= alpha * t
    - || Lw ||_2 <= t       (L = chol(Sigma))
    - sum(w) = 1
    - w in [min_weight, max_weight]

    feasible면 (True, w.value) 반환
    infeasible면 (False, None) 반환
    """
    n = len(mu)
    w = cp.Variable(n)
    t = cp.Variable(nonneg=True)

    # 초과 수익률 벡터
    excess_returns = mu - risk_free_rate
    #print(f"Excess Returns: {excess_returns}")
    # Sigma^(1/2) = L (Cholesky)
    L = np.linalg.cholesky(Sigma)

    constraints = [
        # Sharpe ratio >= alpha -> (mu-rf)^T w >= alpha * t
        excess_returns @ w >= alpha * t,

        # 2차원쌍대제약: ||L w||_2 <= t
        cp.norm(L @ w, 2) <= t,

        # 합 = 1
        cp.sum(w) == 1,

        # 최소/최대 비중
        w >= min_weight,
        w <= max_weight
    ]

    # 목적함수는 단순 "feasibility check"이므로 아무거나 사용
    obj = cp.Maximize(0)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=False)  # 또는 cp.ECOS

    if prob.status in ["optimal", "optimal_inaccurate"]:
        return True, w.value
    else:
        return False, None


def _optimize_mean_variance(mu, Sigma, risk_free_rate=0.03, lam=1.0,
                            min_weight=0.01, max_weight=0.20):
    """
    Mean-Variance(Markowitz) 스타일:
    maximize (mu - rf)^T w - lam * w^T Sigma w
    """
    n = len(mu)
    w = cp.Variable(n)

    excess_returns = mu - risk_free_rate
    print(f"Excess Returns: {excess_returns}")
    objective = cp.Maximize(
        excess_returns @ w - lam * cp.quad_form(w, Sigma)
    )

    constraints = [
        cp.sum(w) == 1,
        w >= min_weight,
        w <= max_weight
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    if w.value is None:
        # 최적화 실패
        return np.zeros(n)
    else:
        return w.value


def _compute_sharpe_ratio(w, mu, Sigma, risk_free_rate=0.03):
    """
    주어진 해 w에 대해 Sharpe Ratio를 계산
    SR = (mu - rf)^T w / sqrt(w^T Sigma w)
    """
    excess_returns = mu - risk_free_rate
    numerator = excess_returns @ w
    denominator = np.sqrt(w @ (Sigma @ w))
    if denominator <= 1e-10:
        return 0.0
    return numerator / denominator

