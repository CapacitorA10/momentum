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
        연결재무제표(CFS) 우선 시도 후, 데이터 없을 경우 재무제표(OFS)로 재시도
        """

        financial_data = None
        for fs_div in ['CFS', 'OFS']:  # CFS 먼저 시도, 없으면 OFS 시도
            print(
                f"bsns_year: {bsns_year}, reprt_code: {reprt_code}, fs_div: {fs_div} 사업보고서(1년): 11011, 반기보고서:11012, 1분기보고서:11013, 3분기보고서:11014")

            url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
            params = {
                'crtfc_key': self.dart_api_key,
                'corp_code': corp_code,
                'bsns_year': bsns_year,
                'reprt_code': reprt_code,
                'fs_div': fs_div
            }
            resp = None
            for attempt in range(3):
                try:
                    resp = requests.get(url, params=params, timeout=10)
                    resp.raise_for_status()
                    break
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching financial data for {corp_code} (fs_div: {fs_div}): {e}, retrying...")
                    time.sleep(1)
            if resp is None:
                print(f"Failed to fetch data for {corp_code} (fs_div: {fs_div}) after multiple retries.")
                continue  # 다음 fs_div 시도

            data = resp.json()
            if data.get('status') != '000':
                print(f"Open DART API Error (fs_div: {fs_div}): {data.get('message')}")
                continue  # 다음 fs_div 시도

            items = data.get('list', [])
            if not items:  # list가 비어있다면 다음 fs_div 시도
                continue

            financial_data = {}
            for item in items:
                account_nm = item.get('account_nm', '').strip()
                sj_div = item.get('sj_div')
                thstrm_amount = item.get('thstrm_amount', None)
                try:
                    thstrm_amount = float(thstrm_amount) if thstrm_amount else 0.0
                except ValueError:
                    thstrm_amount = 0.0

                if sj_div == 'IS':
                    #print(f"account_nm: {account_nm}, thstrm_amount: {thstrm_amount}")
                    if account_nm == '수익(매출액)':
                        financial_data['매출액'] = thstrm_amount
                    elif account_nm == '영업이익(손실)':
                        financial_data['영업이익'] = thstrm_amount
                    elif account_nm == '당기순이익(손실)':
                        financial_data['당기순이익'] = thstrm_amount
                elif sj_div == 'BS':
                    if account_nm == '자본총계':
                        financial_data['기말자본총계'] = thstrm_amount
                    elif account_nm == '자본금':
                        financial_data['자본금'] = thstrm_amount
            break  # CFS 혹은 OFS에서 데이터를 가져왔으면 반복문 탈출

        # 이전 연도 데이터 가져오기 (평균 자본 계산을 위해)
        if financial_data and int(bsns_year) == self.start_year:  # financial_data가 None이 아닌 경우에만 이전년도 데이터 요청
            prev_bsns_year = str(int(bsns_year) - 1)
            prev_year_data = self.get_financial_data(corp_code, prev_bsns_year, reprt_code)

            if prev_year_data and '기말자본총계' in prev_year_data:
                financial_data['기초자본총계'] = prev_year_data['기말자본총계']
                financial_data['평균자본총계'] = (financial_data['기말자본총계'] + financial_data['기초자본총계']) / 2

            # ROE 계산 (평균 자본 사용)
            if '당기순이익' in financial_data and '평균자본총계' in financial_data and financial_data['평균자본총계'] != 0:
                financial_data['ROE'] = financial_data['당기순이익'] / financial_data['평균자본총계'] * 100
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
                    print(f"year: {year}, reprt_code: {reprt_code}")
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
