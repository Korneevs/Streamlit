import sys
sys.path.append('../Lib')
import os

import getpass
import pandas as pd
import numpy as np
import requests
import datetime
import re
import string
import warnings
from tqdm import tqdm_notebook
import bxtools.vertica as vertica
import bxtools.clickhouse as clickhouse
import media.utils as utils
from pathlib import Path
import os
import inspect
import clickhouse_connect
from copy import deepcopy
# from media.m42DatasetWorker.regions import regions as REGIONS
from tqdm.notebook import tqdm
from media.DatasetWorker.QuerySliceInfo import QuerySliceInfo


class execSQLError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
        

class execSQLWorker():
    
    def __init__(self, engine) -> None:
        """
            Инициализация класса
        """
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_columns', None)
        self.engine = engine            
            

    def reset(self) -> None:
        """
            Очистка класса от всех его параметров
        """

        pass
            

    def get_dataset_sql_query(self, sql_file_name) -> str:
        path = os.getcwd() + '/' + sql_file_name

        return utils.read_sql_file(os.path.abspath(path))
    
    
    def check_columns(self, df_columns, sql_file_name: str, query_slice: QuerySliceInfo) -> None:
        df_columns = set(df_columns)
        need_columns = ['region', 'date', 'group_name']
        metrics = query_slice.metrics_list
        max_columns = ['city'] + metrics + need_columns
        if len(df_columns - set(max_columns)) > 0:
            bad_columns = df_columns - set(max_columns)
            raise execSQLError(f"В таблице кастомного SQL запроса {sql_file_name} есть лишние столбцы: {bad_columns}. Их надо убрать.")
        
        if len(set(need_columns) - df_columns) > 0:
            bad_columns = set(need_columns) - df_columns
            raise execSQLError(f"в таблице кастомного SQL {sql_file_name} нет обязательных столбцов {bad_columns}.")
            
        if len(set(metrics) & df_columns) == 0:
            raise execSQLError(f"В таблице кастомного SQL {sql_file_name} нет ни одной необходимой метрики {metrics}!")


    def check_group(self, df: pd.DataFrame, sql_file_name: str) -> None:
        values = set(df['group_name'])
        if len(set(values) - set(['test', 'control'])) > 0:
            bad_values = set(values) - set(['test', 'control'])
            raise execSQLError(f"Плохие значения в столбце group_name в таблице {sql_file_name}: нет обязательных столбцов {bad_values}.")
        
    
    def check_regions(self, df: pd.DataFrame, sql_file_name: str, query_slice: QuerySliceInfo) -> None:
        control = set(query_slice.control_regions)
        test = set(query_slice.test_regions)
        excl = set(query_slice.exclude_regions)
        
        regions = set(df['region'])
        test_regs_m42 = set(df[df['group_name'] == 'test']['region'])
        control_regs_m42 = set(df[df['group_name'] == 'control']['region'])
        
        bad_control = control - regions
        bad_test = test - regions
        
        bad_added_test =  test_regs_m42 - test
        
        if len(bad_control) > 0:
            raise execSQLError(f"Не все регионы контроля есть в датасете {sql_file_name}: {bad_control}")
        
#         print(excl)
        bad_control_2 = control_regs_m42 - control - excl
        bad_control_3 = control - control_regs_m42
        if len(bad_control_2) > 0:
            raise execSQLError(f"В контрольных регионах в SQL {sql_file_name} указан: {bad_control_2} из теста")
            
        if len(bad_control_3) > 0:
            raise execSQLError(f"В контрольных регионах main_params указаны {bad_control_3}, но их нет в SQL")
            
        if len(bad_added_test) > 0:
            raise execSQLError(f"Регионы {bad_added_test} обозначены как тест в {sql_file_name}, но в параметрах JSON-а на вход их нет.")
            
        control_city = set(df[df['group_name'] == 'control']['city'])
        assert len(set(control_city) - set(['Any'])) == 0
    
    
    def check_dataset(self, df: pd.DataFrame, sql_file_name: str, query_slice: QuerySliceInfo) -> None:
        if len(df) == 0:
            raise execSQLError(f"Пустой датасет, полученный {sql_file_name}!")
        
        self.check_columns(df.columns, sql_file_name, query_slice)
        self.check_group(df, sql_file_name)
        self.check_regions(df, sql_file_name, query_slice)
        
    
    
    def get_dataset(self, query_slice: QuerySliceInfo) -> pd.DataFrame:
        sql_file_name = query_slice.data_source_string
        sql = self.get_dataset_sql_query(sql_file_name)
        self.engine.reconnect()
        df = self.engine.select(sql)
        df.fillna(0, inplace=True)
        if 'city' not in df.columns:
            df['city'] = 'Any'
        self.check_dataset(df, sql_file_name, query_slice)
        df['slice_name'] = query_slice.slice_name
        df['date'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d') if not type(x) == str else x)
        return df