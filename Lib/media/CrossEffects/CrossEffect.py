import sys
sys.path.append('../Lib')

import getpass
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
from bxtools.vertica import VerticaEngine
import inspect
import media.utils as utils
import os
import matplotlib.pyplot as plt


class CrossEffect():
    
    
    def __init__(self, engine) -> None:
        """
            Инициализация класса
        """
        self.raw_dataframe = False
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_columns', None)
        self.engine = engine
    
    
    def get_NOT_filled_sql_query(self, file_name) -> str:
        path = "/".join(inspect.getfile(self.__class__).split('/')[:-1])
        path = path + f'/SQL/{file_name}.sql'
        return utils.read_sql_file(os.path.abspath(path))
    
    
    def get_fill_query(self, file_name, params):
        sql_q = self.get_NOT_filled_sql_query(file_name)        
        return utils.fill_query(sql_q, params)
    
    
    def execute_query(self, file_name, params):
        sql = self.get_fill_query(file_name, params)
        sql_parts = sql.split(';')
        for sql in sql_parts:
            sql = sql.strip()
            if len(sql) > 0:
                self.engine.execute(sql)
        return


    def get_params(self, percent=5):
        delta_period = 45
#         end_date = (datetime.datetime.now() - timedelta(2)).strftime("%Y-%m-%d")
        end_date = pd.to_datetime('2024-12-01')
        start_date = (end_date - timedelta(2 * delta_period)).strftime("%Y-%m-%d")
        end_date = (end_date - timedelta(delta_period)).strftime("%Y-%m-%d")
        return {
            'slice_name': 'vertical',
            'start_date': start_date,
            'end_date': end_date,
            'delta_period': delta_period,
            'percent': percent
        }


    def get_precalc_tables_for_crom(self, params):
        self.engine.reconnect()
#         if not force_from_the_beginning:
#             try:
#                 self.engine.select('SELECT * FROM public.ce_buyer_table LIMIT 1')
#                 return
#             except Exception:
#                 pass 
        print('started!')
#         self.execute_query('0_perf_start_session', params)
#         self.execute_query('1_weight_table', params)
#         self.execute_query('2_buyer_table', params)
        print('ended!')
            
    
    def get_data_for_crosseffect(self):
        params = self.get_params(percent=10)        
        
        if isinstance(self.engine, VerticaEngine): 
            self.execute_query('3_perf_effect', params)
            perf_df = self.engine.select("""SELECT * FROM perf_effect""")
            
            params['slice_name'] = 'vertical'
            self.execute_query('4_spillover', params)
            spillover_vertical_df = self.engine.select('SELECT * FROM spillover_df')
            
            params['slice_name'] = 'logical_category'
            self.execute_query('4_spillover', params)
            spillover_logical_df = self.engine.select('SELECT * FROM spillover_df')
        else:
            params['schema'] = f'_u_{self.engine.conn_info["user"]}'
            self.execute_query('3_perf_effect_trino', params)
            perf_df = self.engine.select(f"""SELECT * FROM {params['schema']}.perf_effect""")
            
            params['slice_name'] = 'vertical'
            self.execute_query('4_spillover_trino', params)
            spillover_vertical_df = self.engine.select(f"""SELECT * FROM {params['schema']}.spillover_df""")
            
            params['slice_name'] = 'logical_category'
            self.execute_query('4_spillover_trino', params)
            spillover_logical_df = self.engine.select(f"""SELECT * FROM {params['schema']}.spillover_df""")
            
        self.data_dict = {
            "spillover_vertical_df": spillover_vertical_df,
            "spillover_logical_df": spillover_logical_df,
            "perf_df": perf_df,
        }
        return self.data_dict
    
    
    def make_spillover_effect(self, start_table, target_categories, slice_name='logical_category'):
        df = start_table.copy()
        metric_col = 'w_buyer'

        df = df.groupby([f'start_{slice_name}', slice_name])[metric_col].sum().reset_index()
        metric_dict = df.set_index([f'start_{slice_name}', slice_name]).to_dict()[metric_col]

        den = 0
        num = 0
        columns = sorted(df[slice_name].unique())
        for s_cat in target_categories:
            for spill_cat in columns:
                if (s_cat, spill_cat) in metric_dict and not pd.isnull(metric_dict[(s_cat, spill_cat)]):
                    if spill_cat in target_categories:
                        den += metric_dict[(s_cat, spill_cat)]
                    num += metric_dict[(s_cat, spill_cat)]
        return num / den
    
    
    def get_logcat_to_vertical_coeff(self, logcat_table, vertical_table, target_categories):
        logcat_coeff = self.make_spillover_effect(logcat_table, target_categories)
        vertical_coeff = self.make_spillover_effect(vertical_table, 
                                                    [target_categories[0].split('.')[0]], 
                                                    slice_name='vertical')
        return logcat_coeff / vertical_coeff


    def get_square_perf_coeff(self, perf_df, target_categories):
        vertical = target_categories[0].split('.')[0]
        
        metric_col = 'w_buyer'
        num = perf_df[perf_df['promo_vertical'] == vertical]
        den = perf_df[(perf_df['promo_vertical'] == vertical) & (perf_df['vertical'] == vertical)]

        num = num.groupby('day_ind')[metric_col].sum().reset_index().sort_values('day_ind').reset_index()
        den = den.groupby('day_ind')[metric_col].sum().reset_index().sort_values('day_ind').reset_index()

        organic_num = num[num['day_ind'] > 30][metric_col].mean()
        num['effect'] = num[metric_col] - organic_num

        organic_den = den[den['day_ind'] > 30][metric_col].mean()
        den['effect'] = den[metric_col] - organic_den

        perf_crosseffect = num['effect'].sum() / den['effect'].sum()
#         plt.plot(num['effect'])
#         plt.plot(den['effect'])
        return perf_crosseffect
        
    
    def get_crosseffect_with_calculated_data(self, data_dict, target_categories):
        perf_crosseffect = self.get_square_perf_coeff(data_dict['perf_df'], target_categories)
        v2l_coeff = self.get_logcat_to_vertical_coeff(data_dict['spillover_logical_df'], 
                                                               data_dict['spillover_vertical_df'], target_categories)
        
        spillover_vertical = vertical_coeff = self.make_spillover_effect(data_dict['spillover_vertical_df'], 
                                                    [target_categories[0].split('.')[0]], 
                                                    slice_name='vertical')
#         print(f"perf_crosseffect = {perf_crosseffect}")
#         print(f"v2l_coeff = {v2l_coeff}")
#         print(f"spillover_vertical = {spillover_vertical}")
        if 'Transport' in target_categories[0]:
            return 2.1
        return perf_crosseffect * v2l_coeff