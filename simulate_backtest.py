# simulate_backtest.py

from data_import import DataImporter
from strategy_modules import FactorCalculator, optimize_portfolio, calculate_2years_return
import pandas as pd
import json
import os
from datetime import datetime, timedelta

## 백테스트 초기 셋업
config_path = 'config.json'
data_importer = DataImporter(config_path=config_path)

START_DATE = datetime(2018, 5, 15)
END_DATE = datetime(2024, 12, 30)

results = []

## 백테스팅 진행