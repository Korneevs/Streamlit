from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.integrate import simpson
from scipy.special import expit


class SynteticEffectCreator():
    
    
    def __init__(self):
        pass
    
    
    def create_synt_effect(self, dataset, synt_effect, start_date, interval_length, inflection=14):
        
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        for _, dataset_slice in dataset['ml_datasets'].items():
            target = dataset_slice['target components']
            dataset_ts = dataset_slice['dataset'].copy()
            dates = pd.to_datetime(dataset_ts['date'], format="%Y-%m-%d")
            affected_rows = (dates >= start_date) & (dates < start_date + timedelta(interval_length))
            for col in target:
                cumulative_effect = dataset_ts.loc[affected_rows, col].sum(axis=0)
                uplift = cumulative_effect * (synt_effect - 1)
                daily_uplift = self._distribute_uplift(uplift, interval_length, inflection)
                dataset_ts.loc[affected_rows, col] = dataset_ts.loc[affected_rows, col] + daily_uplift
            dataset_slice['dataset']  = dataset_ts
        return dataset
    
    
    def _distribute_uplift(self, uplift, interval_length, inflection):
        shape = self._create_shape(inflection, interval_length)
        effect = uplift * shape / shape.sum()
        return effect
    
    
    def _create_shape(self, inflection, end):
        days = np.arange(0, end, 1)
        shape = expit(days - inflection)
        return shape