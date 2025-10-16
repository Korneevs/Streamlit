import sys
sys.path.append('../Lib')

import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm_notebook
from copy import deepcopy
from media.DatasetWorker.regions import regions as REGIONS
from tqdm.notebook import tqdm
from media.DatasetWorker.M42DatasetCreator import M42DatasetCreator 
from media.DatasetWorker.execSQLWorker import execSQLWorker
from media.ConfigHandling.RegionsRetrieval import RegionsRetrieval
from media.ConfigHandling.RegionsRetrieval import RegionsRetrieval
from media.ConfigHandling.SlicesRetrieval import GeneralSlicesRetrieval, MediaSlicesRetrieval
from media.ConfigHandling.MetaInfo import MetaInfo
from media.DatasetWorker.ConfigToQuerySlice import ConfigToQuerySlice, MediaConfigToQuerySlice
from media.DatasetWorker.QuerySliceInfo import QuerySliceInfo
from media.DatasetWorker.drop_regions import drop_regions as DROP_REGIONS
from typing import List



class DatasetWorkerError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
        
class DatasetCreator():
    
    def __init__(self, ch_engine, vertica_engine, for_media=True) -> None:
        """
            Инициализация класса
        """
        self.raw_dataframe = False
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_columns', None)
        self.ch_engine = ch_engine
        self.vertica_engine = vertica_engine
        
        self.for_media = for_media #TODO: if possible find a way to do better
        # user may forget it was initialised to work with media
        
        #Class dependencies
        self.m42 = M42DatasetCreator(ch_engine)
        self.exSQL = execSQLWorker(vertica_engine)
        if for_media:
            self.slicesRetrievalMethod = MediaSlicesRetrieval.get_all_slices_and_info_from_config_with_dtb
            self.metaRetrieval = MediaSlicesRetrieval.get_meta_info
            self.configToQuerySlice = MediaConfigToQuerySlice
        else:
            self.slicesRetrievalMethod = GeneralSlicesRetrieval.get_all_slices_and_info_from_config
            self.metaRetrieval = GeneralSlicesRetrieval.get_meta_info
            self.configToQuerySlice = ConfigToQuerySlice
        
        

    def reset(self) -> None:
        """
            Очистка класса от всех его параметров
        """

        pass


    def get_df(self, query_slice: QuerySliceInfo):
        info_type = query_slice.data_source_type
        if info_type == 'm42':
            df = self.m42.get_dataset(query_slice)
        elif info_type == 'sql file':
            df = self.exSQL.get_dataset(query_slice)
        else:
            raise DatasetWorkerError(f"Неправильно задан {query_slice.data_source_string}")
        df = self.leave_only_good_regions(df, query_slice)
        return df


    def leave_only_good_regions(self, df, query_slice):
        control = set(query_slice.control_regions)
        test = set(query_slice.test_regions)
        if 'Any' in test:
            return df
        df = df[df['region'].isin(control | test)]
        return df


    def get_regions_of_slices(self, slice_dataset, group_name):
        df = slice_dataset[slice_dataset['group_name'] == group_name]
        slice_test_regions = df['location'].unique()
        return slice_test_regions


    def get_pivot_datasets(self, dataset: pd.DataFrame, covariates: dict, 
                           query_slice: QuerySliceInfo) -> dict:
        features_json = {}
        slice_name = dataset['slice_name'].unique()[0]
        metrics = dataset.drop(columns=['date', 'slice_name',
                                        'group_name', 'location']).columns
        
        money_df = None
        for metric in metrics:
            test_locs = self.get_regions_of_slices(dataset, 'test')
            features = self.get_regions_of_slices(dataset, 'control')
            cov_dataset = covariates.get(metric)
            
            main_dataset = pd.pivot_table(
                    dataset, values=metric, 
                    index=['date'], columns=['location'], 
                    aggfunc=np.sum, fill_value=0
            ).reset_index()
            
            if cov_dataset is not None:
                print(cov_dataset)
                main_dataset = pd.merge(main_dataset, cov_dataset, on=['date'], how='left')
                features += cov_dataset.drop(columns=['date']).columns
            
            if metric in ['classified_amount_net_adj', 'revenue']:
                money_df = pd.DataFrame(main_dataset[test_locs].sum(axis=1), columns=['revenue'])
                money_df['date'] = main_dataset['date']
            if metric in query_slice.metrics_list:
                features_json[(metric, slice_name)] = {
                    'target components': test_locs,
                    'features': features,
                    'dataset': main_dataset,
                }
                if self.for_media:
                    features_json[(metric, slice_name)] |= {
                    'vertical': query_slice.vertical,
                    'logical_category': query_slice.logical_category,
                    'analysed_budget': query_slice.budget
                    }
        if money_df is not None:
            for key in features_json:
                features_json[key]['money_df'] = money_df
            
        return features_json
    
    
    def check_values(self, slice_dict, value):
        slice_names = set(slice_dict[value])
        good_names = set(self.ch_engine.select(f'SELECT value from dct.m42_{value}')['value'])
        if len(set(slice_names) - set(good_names)) > 0:
            bad = set(slice_names) - set(good_names)
            raise DatasetWorkerError(f"Bad {value} names: {bad}")
        
    


    def get_one_slice_dataset(self, query_slice: QuerySliceInfo) -> dict:
        
        target_df = self.get_df(query_slice)
        target_df = target_df[~target_df['region'].isin(DROP_REGIONS)]
        
        #print(target_df)
        target_df['location'] = target_df.apply(lambda row: self._make_location(row), axis=1)
        target_df.drop(columns=['region', 'city'], inplace=True)
        
#         analyse_date = main_info['max_date']
#         if analyse_date not in set(target_df['date']):
#             raise DatasetWorkerError(f"Даты анализа {analyse_date} нет в полученном датасете. Проверьте дату!")
        
        covariates = {}

        
        for metric in query_slice.metrics_list:
            cov_df = None
            # Not used
            if query_slice.features is not None:
                cov_df = self.get_cov_df(query_slice.slice_name, features[metric])
                assert len(cov_df) == len(cov_df['date'].unique())
            ###
            covariates[metric] = cov_df

        features_json = self.get_pivot_datasets(target_df, covariates, query_slice)
        return features_json
    

    def _make_location(self, row):
        if row['city'] == 'Any':
            return row['region']
        return row['region'] + ": " + row['city']

    
    def get_datasets(self, main_params, slices_json):
         # end_date, test, control, exclude, metrics
        full_dataset_json = {}
        
        config_slices_info_list = self.slicesRetrievalMethod(
            main_params=main_params, slices_json=slices_json, ch_engine=self.ch_engine)
        # get info which is true for all slices:
        meta_info = self.metaRetrieval(main_params)
        
        # validates that metric exists in m42 and not ratio:
        for config_slice in config_slices_info_list:
            config_slice.validate_attributes(self.ch_engine)

        # creates query slices from config slices (may be easy to get data for several config slices with one query):
        query_slices = self.configToQuerySlice.get_query_slices_from_config_slices(
            config_slices=config_slices_info_list) 

        for query_slice in query_slices:
            curr_json = self.get_one_slice_dataset(query_slice)
            full_dataset_json = {**full_dataset_json, **curr_json}
            
        curr_date = list(full_dataset_json.values())[0]['dataset']['date'].max()
        curr_date = datetime.datetime.strptime(curr_date, '%Y-%m-%d') - datetime.timedelta(1)
        curr_date = curr_date.strftime('%Y-%m-%d')
#         if main_params['analysed_end_date'] < main_params['start_date']:
#             raise DatasetWorkerError(f"""Дата анализа не может быть раньше даты старта флайта!""")
#         if curr_date < main_params['analysed_end_date']:
#             main_params['analysed_end_date'] = curr_date
#         if curr_date < main_params['start_date']:
#             raise DatasetWorkerError(f"""Дата старта не может быть больше текущей даты""")
#         if curr_date < main_params['flight_end_date']:
#             main_params['flight_end_date'] = curr_date
        return {
            'ml_datasets': full_dataset_json,
            'metrics': meta_info.metrics,
            "start_date": meta_info.start_date,
            "end_date": meta_info.end_date,
            "flight_end_date": meta_info.flight_end_date,
            "label": meta_info.label
        }
    
    
    # For pipeline
    def get_datasets_from_query_slices(self, query_slices: List[QuerySliceInfo], meta_info: MetaInfo):
        full_dataset_json = {}
        for query_slice in query_slices:
            curr_json = self.get_one_slice_dataset(query_slice)
            full_dataset_json = {**full_dataset_json, **curr_json}
        return {
            'ml_datasets': full_dataset_json,
            'metrics': meta_info.metrics,
            "start_date": meta_info.start_date,
            "end_date": meta_info.end_date,
            "flight_end_date": meta_info.flight_end_date,
            "label": meta_info.label
        }
    