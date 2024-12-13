# data_import.py

import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
import json
import os
import datetime
import time


class DataImporter:
    def __init__(self, config_path='config.json'):
        # config.json 로드
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.dart_api_key = config['DART_API_KEY']
        self.start_year = config['DATA']['start_year']
        self.end_year = config['DATA']['end_year']
        self.start_date = config['DATA']['start_date']
        self.end_date = config['DATA']['end_date']

        # corp_code 매핑 캐시
        self.corp_code_map = self.get_corp_code_map()

    def get_kospi200_list(self):
        """
        네이버 금융에서 KOSPI200 종목 리스트 스크래핑
        """
        url = "https://finance.naver.com/sise/entryJongmok.naver?code=KPI200"
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, 'html.parser')

        # 종목 테이블 파싱
        table = soup.find('table', class_='type_1')
        rows = table.find_all('tr')

        codes = []
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 2:
                name_col = cols[0].find('a')
                if name_col:
                    stock_name = name_col.text.strip()
                    # 종목코드 추출 (네이버금융 종목 URL에서 코드 추출)
                    href = name_col.get('href')
                    # href 예: '/item/main.naver?code=005930'
                    if 'code=' in href:
                        code = href.split('code=')[1]
                        codes.append((stock_name, code))

        df = pd.DataFrame(codes, columns=['Name', 'Code'])
        return df

    def get_corp_code_map(self):
        """
        OpenDART API를 통해 상장회사 목록을 받아 종목코드와 corp_code 매핑
        """
        url = f"https://opendart.fss.or.kr/api/list.json?crtfc_key={self.dart_api_key}&corp_cls=Y"
        response = requests.get(url)
        data = response.json()

        if data['status'] != '013':
            raise Exception(f"Error fetching corp list: {data.get('message', 'Unknown error')}")

        corp_list = data['list']
        corp_code_map = {}
        for corp in corp_list:
            corp_code_map[corp['stock_code']] = corp['corp_code']

        return corp_code_map

    def get_financial_data(self, corp_code, bsns_year, reprt_code):
        """
        OpenDART API를 이용해 해당 기업의 특정 사업연도와 보고서 코드에 대한 재무데이터 수집
        bsns_year: 사업연도 (예: 2020)
        reprt_code: 보고서 코드 (11011: 1분기보고서, 11012: 2분기보고서, 11013: 3분기보고서, 11014: 4분기보고서)
        """
        url = f"https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
        params = {
            'crtfc_key': self.dart_api_key,
            'corp_code': corp_code,
            'bsns_year': bsns_year,
            'reprt_code': reprt_code
        }
        resp = requests.get(url, params=params)
        data = resp.json()

        if data['status'] != '000':
            print(
                f"Error fetching financial data for corp_code {corp_code}, year {bsns_year}, report {reprt_code}: {data.get('message', 'Unknown error')}")
            return None

        # 필요한 항목 추출
        items = data.get('list', [])
        financial_data = {}
        for item in items:
            account_nm = item['account_nm']
            thstrm_amount = float(item['thstrm_amount']) if item['thstrm_amount'] else 0.0
            if account_nm in ['매출액', '영업이익', '당기순이익', '자본금']:
                financial_data[account_nm] = thstrm_amount

        # ROE 계산을 위해 자기자본 추가
        if '당기순이익' in financial_data and '자본금' in financial_data:
            financial_data['ROE'] = financial_data['당기순이익'] / financial_data['자본금'] * 100
        else:
            financial_data['ROE'] = None

        return financial_data

    def get_all_financial_data(self, kospi200_df):
        """
        KOSPI200 종목 전체에 대한 재무데이터 수집
        """
        financial_records = []
        for index, row in kospi200_df.iterrows():
            stock_code = row['Code']
            corp_code = self.corp_code_map.get(stock_code)
            if not corp_code:
                print(f"Corp code not found for stock code: {stock_code}")
                continue

            for year in range(self.start_year, self.end_year + 1):
                for reprt_code in ['11011', '11012', '11013', '11014']:
                    financial = self.get_financial_data(corp_code, year, reprt_code)
                    if financial:
                        financial['Code'] = stock_code
                        financial['Year'] = year
                        financial['Report'] = reprt_code
                        financial_records.append(financial)
                    time.sleep(0.1)  # API 호출 제한 대비
        financial_df = pd.DataFrame(financial_records)
        return financial_df

    def get_price_data(self, tickers):
        """
        yfinance를 통해 일별 주가 다운로드
        """
        data = yf.download(tickers, start=self.start_date, end=self.end_date)
        # data['Adj Close']가 MultiIndex('Adj Close', ticker) 형태로 나오므로 이를 정리
        price_df = data['Adj Close']
        return price_df
