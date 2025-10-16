import sys
sys.path.append('../Lib')

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
from media.DatasetWorker.m42_extra_conditions import skip_params_to_check
from media.DatasetWorker.QuerySliceInfo import QuerySliceInfo
from media.DatasetWorker.IdNamesCorrespondenceM42 import IdNamesCorrespondenceM42
# from media.m42DatasetWorker.regions import regions as REGIONS
from tqdm.notebook import tqdm
from urllib.parse import unquote_plus


class M42Error(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
        
        
class M42DatasetCreator():
    
    def __init__(self, engine) -> None:
        """
            Инициализация класса
        """
        self.raw_dataframe = False
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_columns', None)
        self.engine = engine
        self.params_df = self.get_m42_parameters()

        #Class dependencies
        self.idNamesCorrespondenceM42 = IdNamesCorrespondenceM42
            
            

    def reset(self) -> None:
        """
            Очистка класса от всех его параметров
        """

        pass
            

    def get_m42_dataset_NOT_filled_sql_query(self) -> str:
        path = "/".join(inspect.getfile(self.__class__).split('/')[:-1])
        path = path + '/sql/m42_sql_body.sql'

        return utils.read_sql_file(os.path.abspath(path))
    
    
    def get_m42_parameters(self):
        df = self.engine.select(
        """DESCRIBE TABLE dma.m42"""
        )[['name', 'type']]
        df['column_name'] = df['name']
        df['name'] = df['name'].apply(lambda x: x[:-3] if '_id' == x[-3:] else x)
        df = df[(df['column_name'] != 'launch_id') & (~df['name'].apply(lambda x: 'reserved_' in x))]
        
        dicts = set(self.engine.select("SELECT name FROM system.dictionaries WHERE name like 'm42_%'")['name'])
        df['dict'] = df['name'].apply(lambda x: f'dct.m42_{x}' if f'm42_{x}' in dicts else '')
        df = df[~df['column_name'].isin(skip_params_to_check)]
#         print(df, dicts)
        if len(df[df['dict'] == '']) > 0:
            bad = list(df[df['dict'] == '']['column_name'])
            raise M42Error(f"""Не для всех параметров dma.m42 есть dict в кликхаусе: {bad}""")
        return df
    

    def get_params_from_link(self, link):
        link = unquote_plus(link)
        
        params_pattern = r'([^?&]+)' # nonempty text, split by ?, & 
        params = re.findall(params_pattern, link)[1:] # [1:] excludes https://m42.k.avito.ru/
        
        link_dict = {}
        for param in params:
            column, value = param.split('=')
            link_dict[column] = utils.clean_array_from_strings_if_possible(value.split(','))
        return link_dict

    def get_default(self, column):
        if column == 'is_human':
            return [1]
        row = self.params_df[self.params_df['column_name'] == column].iloc[0]
        if 'Int' in row['type']:
            return [-1]
        return ['']

    def parse_m42_link(self, link):
        link_dict = self.get_params_from_link(link)
        condition_json = {}
        for _, row in self.params_df.iterrows():
            full_col = row['column_name']
            col = row['name']
            if col not in link_dict:
                if col == 'is_participant_new':
                    continue 
                link_dict[col] = self.get_default(full_col)
#                 raise M42Error(f"""Не все параметры dma.m42 отражены в ссылке slice-а, а точнее: {col}. 
# Проверьте корректность ссылки.""")
            condition_json[col] = link_dict[col]   
        return condition_json
    
    
#     def check_main_params_and_reorganise(self):
    
    def get_metric_ids_data(self, query_slice_info):
        conditions = []
        metrics = ', '.join([f"'{metric}'" for metric in query_slice_info.metrics_list]) # + ['classified_amount_net_adj']
        
        metric_df = self.engine.select(f"SELECT * FROM dct.m42_metric WHERE metric in ({metrics})")
        
        metric_ids = list(metric_df["id"])
        return metric_ids, metric_df.set_index("id")['metric'].to_dict()


    def _get_group_condition(self, test_region_cond, control_array):
        control = str(control_array)[1:-1]
#         test = str(test_array)[1:-1]
        if len(control) == 0:
            control = -2
            any_id = self.idNamesCorrespondenceM42.regions_to_id(['Any'], self.engine)[0]
            return f"""'test'"""
                
        return f"""multiIf(
            {test_region_cond}, 'test',
            region_id in ({control}), 'control',
            'Undefined'
        )
        """


    def _get_condition_one_param(self, condition_json, key, names_dict):
        value = np.array(condition_json[key])
        value[value == 0] = 2
        value[value == -1] = 0
        value = list(value)
        column = names_dict[key]
        curr_sql = f"{column} in ({str(value)[1:-1]})"
        return curr_sql
        
    
    
    def get_link_condition(self, query_slice_info):

        condition_json = self.parse_m42_link(query_slice_info.data_source_string)
        control_id = self.idNamesCorrespondenceM42.regions_to_id(query_slice_info.control_regions, 
                                                                 self.engine)
#         print(condition_json)
        condition = []
        names_dict = self.params_df.set_index('name')['column_name'].to_dict()
        is_start = True
        test_region_cond = ""
        for key in condition_json:
            if key not in names_dict or key in ('metric', 'city'):
                continue
            
            curr_sql = is_start * "\t\t" + self._get_condition_one_param(condition_json, key, names_dict)
            if key == 'region':
#                print(condition_json)
                city_cond = self._get_condition_one_param(condition_json, 'city', names_dict)
                curr_sql = "(" + curr_sql + " AND " + city_cond + ")"
                test_region_cond = curr_sql
                if len(control_id) > 0:
                    control = str(control_id)[1:-1]
                    curr_sql = "(\n\t\t    " + curr_sql + f" --test\n\t\t OR "\
                        f"(region_id in ({control}) AND city_id in (0)) --control\n\t\t)"
            condition.append(curr_sql)
            is_start = False
        condition = '\t(\n' + "\n\t\tAND ".join(condition) + '\n\t)'
        return condition, test_region_cond


    def _make_metrics_columns(self, id2metric):
        m_columns = []
        for m_id, m_name in id2metric.items():
            col = f"sumIf(metric_value, metric_id == {m_id}) as {m_name},"
            m_columns.append(col)
        return "\n    ".join(m_columns)[:-1]
    
    def _ids_to_regions(self, array):
        reg_dict = self.engine.select(f"SELECT * FROM dct.m42_region").set_index('id')['value'].to_dict()
        return [reg_dict[reg_id] for reg_id in array]


    def get_sql_params(self, query_slice_info):
        slice_column_if = {}
        
        control_id = self.idNamesCorrespondenceM42.regions_to_id(query_slice_info.control_regions, 
                                                                 self.engine)
        test_id = self.idNamesCorrespondenceM42.regions_to_id(query_slice_info.test_regions, 
                                                              self.engine)
        
        condition, test_region_cond = self.get_link_condition(query_slice_info)
#        print(condition)
        
        metric_ids, id2metric = self.get_metric_ids_data(query_slice_info)

        return {
            'all_slices_no_launch_id': ", ".join(self.params_df['column_name']),
            "conditions": "AND (\n" + condition + "\n\t\t)",
            "slice_column": f"'{query_slice_info.slice_name}'",
            "group_column": self._get_group_condition(test_region_cond, control_id),
            "metric_ids": ', '.join([str(id) for id in metric_ids]),
            "metric_columns": self._make_metrics_columns(id2metric)
        }
    
    
    def get_m42_dataset_filled_sql_query(self, query_slice_info):
        sql_not_filled = self.get_m42_dataset_NOT_filled_sql_query()
        
        params = self.get_sql_params(query_slice_info)
#         print(params)
        
        return utils.fill_query(sql_not_filled, params)

    
    def _check_undefind(self, df, column, slice_name, link, main_info):
        if 'Undefined' in list(df[column]) and column not in ['region', 'city']:
            raise M42Error(f"{column}: обнаружилось undefind значение. "\
                  f"Проверьте датасет, полученный скриптом dc.m42.get_m42_dataset_filled_sql_query"\
                  f"""("{slice_name}", "{link}", {main_info})""")

    
    def get_correct_link(self, link):
        #TODO: change correct link
        #link = link.split('state_hash=')[0] + "&" + "&".join(link.split('&')[1:])
        return link
    
        
    def check_dataset(self, df, query_slice):
        slice_name = query_slice.slice_name
        link = query_slice.data_source_string
        main_info = {
            'test_regions': query_slice.test_regions,
            'control_regions': query_slice.control_regions,
            'exclude_regions': query_slice.exclude_regions,
            'start_date': query_slice.start_date,
            'end_date': query_slice.end_date
        }

        if len(df) == 0:
            raise M42Error(f"Пустой датасет! "\
                            f"Проверьте датасет, полученный скриптом dc.m42.get_m42_dataset_filled_sql_query"\
                            f"""("{slice_name}", "{link}", {main_info})""")
        df['region'] = df['region'].replace('Undefined', 'Any')
        df['city'] = df['city'].replace('Undefined', 'Any')
#         self._check_undefind(df, 'region', slice_name, link, main_info)
#         self._check_undefind(df, 'city', slice_name, link, main_info)
        self._check_undefind(df, 'group_name', slice_name, link, main_info)
        
        control = set(query_slice.control_regions)
        test = set(query_slice.test_regions)
        regions = set(df['region'])
        test_regs_m42 = set(df[df['group_name'] == 'test']['region'])
        
        bad_control = control - regions
        bad_test = test - regions
        
        bad_added_test =  test_regs_m42 - test
        if len(bad_control) > 0:
            raise M42Error(f"Не все регионы контроля есть в датасете M42: {bad_control}")
#         if len(bad_test) > 0:
#             raise M42Error(f"Не все регионы теста есть в датасете M42: {bad_test}, проверьте ссылку {self.get_correct_link(link)}")
        if len(bad_added_test) > 0 and set(test) != set(['Any']):
            raise M42Error(f"Добавлены регионы {bad_added_test} в ссылку, но не в параметры."\
                           f" Проверьте ссылку {self.get_correct_link(link)} и параметры")



    def get_dataset(self, query_slice: QuerySliceInfo):
        sql = self.get_m42_dataset_filled_sql_query(query_slice)
#         print(sql)
        df = self.engine.select(sql)
        self.check_dataset(df, query_slice)
        
        df['date'] = df['metric_date'].astype(str)
        df.drop(columns=['metric_date'], inplace=True)
        return df