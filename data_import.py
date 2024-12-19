# data_import.py

import requests
import pandas as pd
import yfinance as yf
import json
import os
import time
import zipfile
import io
import xml.etree.ElementTree as ET


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
        import requests
        from bs4 import BeautifulSoup

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
        OpenDART API를 통해 상장회사 목록(고유번호 파일)을 다운로드한 후
        stock_code -> corp_code 매핑을 생성
        """
        url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={self.dart_api_key}"
        response = requests.get(url)

        if response.status_code != 200:
            raise Exception(f"Error fetching corp code zip: HTTP {response.status_code}")

        # 응답은 ZIP 파일 형태. 메모리에 로드 후 zipfile로 처리
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # ZIP 내부의 CORPCODE.xml 파일을 읽는다.
            # 파일명이 CORPCODE.xml 인지 확인 (실제 제공되는 파일명은 보통 CORPCODE.xml)
            for filename in z.namelist():
                if filename.upper().endswith('.XML'):
                    with z.open(filename) as xml_file:
                        tree = ET.parse(xml_file)
                        root = tree.getroot()
                        # root 밑에 <list> 태그들이 기업 정보
                        corp_code_map = {}
                        for list_item in root.findall('list'):
                            stock_code = list_item.find('stock_code').text.strip()
                            corp_code = list_item.find('corp_code').text.strip()
                            # stock_code가 없는 경우 비상장 회사일 수 있음
                            # 또는 공백인 경우 제외
                            if stock_code and stock_code != ' ':
                                corp_code_map[stock_code] = corp_code
                        print(f"Total corp codes fetched: {len(corp_code_map)}")
                        return corp_code_map

        raise Exception("No XML file found in the downloaded zip.")

    def get_financial_data(self, corp_code, bsns_year, reprt_code):
        """
        OpenDART API를 이용해 해당 기업의 특정 사업연도와 보고서 코드에 대한 재무데이터 수집
        """
        url = f"https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
        params = {
            'crtfc_key': self.dart_api_key,
            'corp_code': corp_code,
            'bsns_year': bsns_year,
            'reprt_code': reprt_code,
            'fs_div': 'CFS'  # CFS: 연결재무제표, OFS: 재무제표
        }
        resp = None
        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                break
            except:
                print(f"Error fetching financial data for {corp_code}, retrying...")
                time.sleep(1)
        data = resp.json()
        if data.get('status') != '000':
            # 공시 없는 경우도 있을 수 있음
            return None

        # 필요한 항목 추출
        items = data.get('list', [])
        financial_data = {}
        for item in items:
            account_nm = item.get('account_nm', '').strip()
            thstrm_amount = item.get('thstrm_amount', None)
            try:
                thstrm_amount = float(thstrm_amount) if thstrm_amount else 0.0
            except ValueError:
                thstrm_amount = 0.0
            if item.get('sj_div') == 'IS':
                if account_nm == '수익(매출액)':
                    financial_data['매출액'] = thstrm_amount
                elif account_nm == '영업이익(손실)':
                    financial_data['영업이익'] = thstrm_amount
                elif account_nm == '당기순이익(손실)':
                    financial_data['당기순이익'] = thstrm_amount
            elif item.get('sj_div') == 'SCE' and item.get('account_nm') == '기말자본' and "자본금 [member]" in item.get(
                    'account_detail', ""):
                financial_data['자본금'] = thstrm_amount

        # ROE 계산
        if '당기순이익' in financial_data and '자본금' in financial_data and financial_data['자본금'] != 0:
            financial_data['ROE'] = financial_data['당기순이익'] / financial_data['자본금'] * 100
        else:
            financial_data['ROE'] = None

        return financial_data

    def get_all_financial_data(self, kospi200_df):
        """
        KOSPI200 종목 전체에 대한 재무데이터 수집
        """
        financial_records = []
        total = len(kospi200_df)
        for idx, row in kospi200_df.iterrows():
            stock_code = row['Code']
            corp_code = self.corp_code_map.get(stock_code)
            if not corp_code:
                print(f"Corp code not found for stock code: {stock_code}")
                continue

            for year in range(self.start_year, self.end_year + 1):
                # 1분기:11013, 반기(2분기):11012, 3분기:11014, 4분기(사업보고서):11011
                # OpenDART 문서에 따라 reprt_code 확인 필요
                # *참고: 사업보고서(1년): 11011, 반기보고서:11012, 1분기보고서:11013, 3분기보고서:11014
                # 여기서는 예시로 1,2,3,4분기 모두 시도
                for reprt_code in ['11013', '11012', '11014', '11011']:
                    financial = self.get_financial_data(corp_code, year, reprt_code)
                    if financial:
                        financial['Code'] = stock_code
                        financial['Year'] = year
                        financial['Report'] = reprt_code
                        financial_records.append(financial)
                    time.sleep(0.7)  # API 호출 제한 대비
            print(f"Processed {idx + 1}/{total} stocks.")
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
