# data_import.py

import requests
import pandas as pd
import yfinance as yf
import json
import os
import time
from datetime import datetime, timedelta
from pykrx import stock

class DataImporter:
    def __init__(self, config_path='config.json'):
        # config.json 로드
        self.config_path = config_path
        self.config = self.load_config()

        self.kis_appkey = self.config['KIS_API']['appkey']
        self.kis_appsecret = self.config['KIS_API']['appsecret']
        self.cano = self.config['KIS_API']['CANO']
        self.acnt_prdt_cd = self.config['KIS_API']['ACNT_PRDT_CD']
        self.start_date = datetime.now() - timedelta(days=365 * 2)
        self.end_date = datetime.now()

        # KOSPI200 종목 리스트 가져오기
        self.kospi200_df = self.get_kospi200_list()

        # Access Token 발급 또는 기존 토큰 사용
        self.kis_access_token = self.get_kis_access_token()

    def load_config(self):
        """
        config.json 파일을 로드하여 반환합니다.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"{self.config_path} not found.")
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_config(self):
        """
        현재 config를 config.json 파일에 저장합니다.
        """
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)

    def get_kis_access_token(self):
        """
        KIS API 인증을 위한 Access Token 발급 또는 기존 토큰 사용
        """
        access_token = self.config['KIS_API'].get('access_token')
        token_issue_time_str = self.config['KIS_API'].get('token_issue_time')

        if access_token and token_issue_time_str:
            token_issue_time = datetime.strptime(token_issue_time_str, "%Y-%m-%dT%H:%M:%S")
            if datetime.now() - token_issue_time < timedelta(hours=24):
                print("기존 Access Token을 사용합니다.")
                return access_token
            else:
                print("Access Token이 만료되었습니다. 새로운 토큰을 발급받습니다.")
        else:
            print("Access Token이 존재하지 않습니다. 새로운 토큰을 발급받습니다.")

        # 새로운 Access Token 발급
        url = "https://openapi.koreainvestment.com:9443/oauth2/tokenP"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.kis_appkey,
            "appsecret": self.kis_appsecret
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            token_info = response.json()
            new_access_token = token_info.get('access_token')
            if not new_access_token:
                raise Exception("Access token not found in the response.")
            # config.json 업데이트
            self.config['KIS_API']['access_token'] = new_access_token
            self.config['KIS_API']['token_issue_time'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            self.save_config()
            print("새로운 Access Token을 발급받아 config.json에 저장했습니다.")
            return new_access_token
        else:
            raise Exception(f"Failed to obtain access token: {response.text}")

    def get_kospi200_list(self):
        """
        pykrx를 사용하여 KOSPI200 종목 리스트를 가져오는 함수
        """
        try:
            # pykrx의 get_index_portfolio 함수를 사용하여 DataFrame 반환
            kospi200 = stock.get_index_portfolio_deposit_file(ticker = "1028")  # "1028"은 KOSPI200의 지수 코드
            # kospi200 이 []인 경우
            if kospi200 == []:
                print("KOSPI200 종목 리스트를 가져오지 못했습니다.")
                return pd.DataFrame()
            # DataFrame 컬럼을 적절히 변경
            kospi200_df = pd.DataFrame(kospi200, columns=['Code'])
            print("KOSPI200 종목 리스트를 성공적으로 가져왔습니다.")
            return kospi200_df
        except Exception as e:
            print(f"KOSPI200 종목 리스트를 가져오는 중 오류 발생: {e}")
            return pd.DataFrame()

    def get_financial_data_income_statement(self, stock_code):
        """
        KIS API를 통해 손익계산서 데이터 조회
        """
        url = ("https://openapi.koreainvestment.com:9443"
               "/uapi/domestic-stock/v1/finance/income-statement")
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.kis_access_token}",
            "appkey": self.kis_appkey,
            "appsecret": self.kis_appsecret,
            "tr_id": "FHKST66430200",
            "custtype": "P",
        }
        params = {
            "FID_DIV_CLS_CODE": "1",  # 분기
            "fid_cond_mrkt_div_code": "J",  # 조건 시장 분류 코드
            "fid_input_iscd": stock_code
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get('rt_cd') == '0':
                return data.get('output', [])
            else:
                print(f"API Error for {stock_code}: {data.get('msg1')}")
                return []
        else:
            print(f"HTTP Error for {stock_code}: {response.status_code}")
            return []

    def get_financial_data_financial_ratio(self, stock_code):
        """
        KIS API를 통해 재무비율 데이터 조회
        """
        url = ("https://openapi.koreainvestment.com:9443/"
               "uapi/domestic-stock/v1/finance/financial-ratio")
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.kis_access_token}",
            "appkey": self.kis_appkey,
            "appsecret": self.kis_appsecret,
            "tr_id": "FHKST66430300",
            "custtype": "P"
        }
        params = {
            "FID_DIV_CLS_CODE": "1",  # 분기
            "fid_cond_mrkt_div_code": "J",  # 조건 시장 분류 코드
            "fid_input_iscd": stock_code
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get('rt_cd') == '0':
                return data.get('output', [])
            else:
                print(f"API Error for {stock_code}: {data.get('msg1')}")
                return []
        else:
            print(f"HTTP Error for {stock_code}: {response.status_code}")
            return []

    def get_all_financial_data(self):
        """
        KOSPI200 종목 전체에 대한 재무데이터 수집
        """
        financial_records = []
        total = len(self.kospi200_df)
        for idx, row in self.kospi200_df.iterrows():
            stock_code = row['Code']
            print(f"Processing {idx + 1}/{total}: {stock_code}")

            # 손익계산서 데이터 가져오기
            income_statements = self.get_financial_data_income_statement(stock_code)
            time.sleep(0.05)
            if not income_statements:
                continue

            # 재무비율 데이터 가져오기
            financial_ratios = self.get_financial_data_financial_ratio(stock_code)
            time.sleep(0.05)
            if not financial_ratios:
                continue

            # 최근 4개 분기 데이터 추출
            num_records = min(4, len(income_statements), len(financial_ratios))
            for i in range(num_records):
                inc = income_statements[i]
                ratio = financial_ratios[i]
                try:
                    record = {
                        'Code': stock_code,
                        'YearMonth': inc.get('stac_yymm', ''),
                        '매출액': float(inc.get('sale_account', '0').replace(',', '')),
                        '영업이익': float(inc.get('bsop_prti', '0').replace(',', '')),
                        'ROE': float(ratio.get('roe_val', '0').replace(',', ''))
                    }
                    financial_records.append(record)
                except ValueError as ve:
                    print(f"ValueError for {stock_code} in record {i}: {ve}")
                    continue

        financial_df = pd.DataFrame(financial_records)
        return financial_df

    def get_price_data(self, tickers):
        """
        yfinance를 통해 일별 주가 다운로드
        """
        data = yf.download(tickers, start=self.start_date, end=self.end_date)
        if isinstance(data.columns, pd.MultiIndex):
            price_df = data['Adj Close']
        else:
            price_df = data
        return price_df

if __name__ == "__main__":
    importer = DataImporter(config_path='config.json')
    financial_df = importer.get_all_financial_data()
    print(financial_df.head())
##

