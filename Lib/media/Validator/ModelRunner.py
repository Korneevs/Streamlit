import sys
lib_dir = '/srv/data/my_shared_data/Lib/media/'
if lib_dir not in sys.path:
    sys.path.append(lib_dir)
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm.notebook import tqdm
import pandas as pd

from media.Validator.PeriodFinder import PeriodFinder
from media.Validator.PanelGenerator import PanelGenerator
from media.Validator.SliceDictParser import SliceDictParser
from media.Validator.SynteticEffectCreator import SynteticEffectCreator
from media.Validator.UtilityDataFetcher import UtilityDataFetcher
from media.DatasetWorker.DatasetCreator import DatasetCreator
from statsmodels.stats.proportion import proportion_confint


class ModelRunner():
    
    
    def __init__(self, engine_c, engine_v, model, alpha, **kwargs):
        self.get_result_dataset_func = partial(model.get_result_dataset, alpha = alpha, **kwargs)
        self.model = model
        self.engine_c = engine_c
        self.engine_v = engine_v
        self.parser = SliceDictParser()
        self.synt_effect_creator = SynteticEffectCreator()
        self.util_data_fetcher = UtilityDataFetcher()
        self.kwargs = kwargs
        self.results = []
        self.panel_run_args = []
        self.saved_args = []

    
    def show_tmp_results(self, df, synt_effect):
        if synt_effect is None:
            effect = 0
        else:
            effect = synt_effect - 1
        fpr = (df['effect left'] > effect).mean() + (df['effect right'] < effect).mean()
        N = len(df)
        l, r = proportion_confint(count = fpr * N, nobs = N, alpha=0.05, method='wilson')
        print(f"size: {N}, FPR: {round(fpr, 4)}, [{round(l, 4)}, {round(r, 4)}]")
        
          
    # TODO: account for differnt stratification strategies in CUPED
    def run_models(self, validation_slices, metric, start, end, 
                   interval_length, prevalidation_clean_days, min_region_group_size,
                   daily_budget_threshold, reference_region='РФ', 
                   federal=True, test_size=None, target_regs=None, 
                   synt_effect=None, multiprocess=True, excluded_intervals=None, sep='; '):
        
        pf = PeriodFinder(self.engine_v)
        pg = PanelGenerator()
        dc = DatasetCreator(self.engine_c, self.engine_v)
        vert_to_id, log_to_id, reg_to_id = self.util_data_fetcher.create_ch_id_dicts(self.engine_c)
        self.results = []
        self.panel_run_args = []
        self.datasets = []
        
        pools = cpu_count()
        run_model = partial(analyze_panel, get_result_dataset_func=self.get_result_dataset_func)
        
        for validation_slice in validation_slices:
            label = validation_slice['label']
            print(f'Running validation in {label}...')
            slice_dict = validation_slice['slice_dict'] 
            target_slice = slice_dict[0]
            parsed_target_slice = self.parser.parse_query_dict(target_slice, sep)
            vertical = parsed_target_slice['vertical']
            log_cats = parsed_target_slice['logical_category']
            if (test_size is not None or min_region_group_size is not None) and target_regs is not None:
                raise ValueError('If a list of target regions is passed, test will consist only of these regions \
                and rest will be in control: test_size and min_region_group_size should be None')
            clean_intervals = pf.get_clean_intervals(data_dicts=slice_dict, start=start, end=end,
                                                     interval_length=interval_length,
                                                     prevalidation_clean_days=prevalidation_clean_days,
                                                     daily_budget_threshold=daily_budget_threshold,
                                                     reference_region=reference_region, target_regs=target_regs,
                                                     excluded_intervals=excluded_intervals, sep=sep)
            if federal:
                val_pans = pg.sample_regions_fed_models(clean_intervals, min_group_size=min_region_group_size)
            else:
                if target_regs is not None:
                    val_pans = pg.sample_regions_reg_models_fixed_test(clean_intervals, target_regs)
                else:
                    val_pans = pg.sample_regions_reg_models(clean_intervals, min_group_size=min_region_group_size, 
                                                      test_size=test_size)
                    #val_pans = pg.sample_with_paired_stratification(clean_intervals, min_region_group_size, 
                    #                                                test_size, vertical, log_cats, metric, 
                    #                                                self.kwargs['learning_period'], self.engine_c) ##### TO REMOVE
                    #val_pans = pg.sample_with_fixed_test_matched_paired_control(clean_intervals, min_region_group_size, 
                                                #test_size, vertical, log_cats, metric, 
                                                #self.kwargs['learning_period'], self.engine_c) #### TO REMOVE
            print(f'Number of validation panels: {len(val_pans)}')
            
            if multiprocess:
                parts = (len(val_pans) - 1) // pools + 1
            else:
                parts = len(val_pans)
            
            for i in tqdm(range(parts)):
                pans_batch = val_pans[i::parts]
                if len(pans_batch):
                    datasets = []
                    for panel in pans_batch:
                        dataset = self.fetch_data(panel, vertical, log_cats, metric,
                                                  dc, vert_to_id, log_to_id, reg_to_id, federal)
#                         if synt_effect:
#                             dataset = self.synt_effect_creator.create_synt_effect(dataset, synt_effect, panel['start_date'],
#                                                                                   interval_length)
#                         self.datasets.append(dataset)
#                         datasets.append(dataset) # Can add custom fetch method
#                     if multiprocess:
#                         pool = Pool(pools)
#                         analyze_res = pool.map(run_model, datasets)   
#                         pool.close()
#                     else:
#                         analyze_res = [run_model(datasets[0])]
#                     additional_data = {'label': label, 'vertical': '; '.join(vertical)} 
#                     self.results.extend([res | additional_data for res_list in analyze_res for res in res_list])
#                     self.show_tmp_results(pd.DataFrame(self.results), synt_effect)
                    #print(len(curr_df), 
                    #      (curr_df['effect left'] > float(synt_effect) - 1).mean() + (curr_df['effect right'] < float(synt_effect) - 1).mean()
                     #    )

#         return pd.DataFrame(self.results)
        return pd.DataFrame(self.saved_args)
        
    
    
    def run_yoy_models(self, validation_slices, metric, start, end, 
                       interval_length, min_region_group_size,
                       period_budget_diff_threshold, reference_region='РФ',  
                       synt_effect=None, multiprocess=True, excluded_intervals=None, sep='; '):
        
        pf = PeriodFinder(self.engine_v)
        pg = PanelGenerator()
        dc = DatasetCreator(self.engine_c, self.engine_v)
        vert_to_id, log_to_id, reg_to_id = self.util_data_fetcher.create_ch_id_dicts(self.engine_c)
        learning_period = self.kwargs['learning_period']
        self.results = []
        
        pools = cpu_count()
        run_model = partial(analyze_panel, get_result_dataset_func=self.get_result_dataset_func)
        
        for validation_slice in validation_slices:
            label = validation_slice['label']
            print(f'Running validation in {label}...')
            slice_dict = validation_slice['slice_dict'] 
            target_slice = slice_dict[0]
            parsed_target_slice = self.parser.parse_query_dict(target_slice, sep)
            vertical = parsed_target_slice['vertical']
            log_cats = parsed_target_slice['logical_category']
            clean_intervals = pf.get_clean_intervals_yoy(data_dicts=slice_dict, start=start, end=end, yoy=self.model,
                                                         interval_length=interval_length, 
                                                         period_budget_diff_threshold=period_budget_diff_threshold, 
                                                         reference_region=reference_region, 
                                                         excluded_intervals=excluded_intervals, sep=sep, 
                                                         **self.kwargs)
                
            print(f'Number of validation panels: {len(val_pans)}')
            
            
            if multiprocess:
                parts = (len(val_pans) - 1) // pools + 1
            else:
                parts = len(val_pans)
            
            for i in tqdm(range(parts)):
                pans_batch = val_pans[i::parts]
                if len(pans_batch):
                    datasets = []
                    for panel in pans_batch:
                        dataset = self.fetch_data(panel, vertical, log_cats, metric,
                                                  dc, vert_to_id, log_to_id, reg_to_id, federal=True)
                        if synt_effect:
                            dataset = self.synt_effect_creator.create_synt_effect(dataset, synt_effect, panel['start_date'],
                                                                                  interval_length)
                        datasets.append(dataset) # Can add custom fetch method
                    if multiprocess:
                        pool = Pool(pools)
                        analyze_res = pool.map(run_model, datasets)   
                        pool.close()
                    else:
                        analyze_res = [run_model(datasets[0])]
                    additional_data = {'label': label, 'vertical': '; '.join(vertical)} 
                    self.results.extend([res | additional_data for res_list in analyze_res for res in res_list])
                    
                        
        return pd.DataFrame(self.results)
    
    
    def fetch_data(self, panel, vertical, log_cats, metric, dc, vert_to_id, log_to_id, reg_to_id, federal):
        main_params = {
            'label': 'testing',
            "start_date": panel['start_date'],
            "flight_end_date": panel['end_date'],
            "analysed_end_date":  panel['end_date'],
            'metrics': [metric], # 
            "exclude regions": [],
            "source": "M42"}
        
        if federal:
            main_params = main_params | { 
                    "test regions": [reg for reg in panel['regions'] if reg in reg_to_id],
                    "control regions": []}
        else:
            main_params = main_params | {
                    "test regions": [reg for reg in panel['test_regions'] if reg in reg_to_id],
                    "control regions": [reg for reg in panel['control_regions'] if reg in reg_to_id]}
        
        log_ids = ','.join([str(log_to_id[log]) for log in log_cats if log in log_to_id])
        reg_ids = ','.join([str(reg_to_id[reg]) for reg in main_params['test regions'] if reg in reg_to_id])
        vertical_id = ','.join([str(vert_to_id[ver]) for ver in vertical if ver in vert_to_id])
        
        slices = f"https://m42.k.avito.ru/?&logical_category={log_ids}&metric=1792&region={reg_ids}&report=main&sum_by=logical_category,region&vertical={vertical_id}" # main params metric overrides 'metric=1792'
        slices_json = {
            "Any": {'target': slices,
                    'vertical': vertical,
                    'logical_category': log_cats}
        }
        curr_arg = {
            'start_date':  panel['start_date'],
            'end_date': panel['end_date'],
            'vertical': vertical,
            'logical_category': log_cats,
            'test_regs': main_params["test regions"],
            'control_regs': main_params["control regions"],
        }
        self.saved_args.append(curr_arg)
        self.panel_run_args.append((main_params, slices_json))
        dataset_result = None # dc.get_datasets(main_params, slices_json)
        return dataset_result
    

def analyze_panel(dataset_result, get_result_dataset_func):
    ts_df, _ = get_result_dataset_func(dataset_result)
    return get_effects_and_ci(ts_df)


def get_effects_and_ci(ts_df):
    end = ts_df['end_date']
    df = ts_df[ts_df['date'] == end]
    res = []
    for _, row in df.iterrows():
        res.append({'slice_name': row['slice_name'], 
                    'metric': row['metric'], 
                    'effect': row[f'rel. effect'], 
                    'effect left': row[f'rel. CI left'], 
                    'effect right': row[f'rel. CI right'],
                    'start_date': row['start_date'],
                    'end_date': row['end_date']})
    return res