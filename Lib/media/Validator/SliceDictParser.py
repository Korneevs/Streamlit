import pandas as pd
from datetime import datetime


class SliceDictParser():
    
    
    def __init__(self):
        pass
    
    
    def parse_query_dict(self, data_dict, sep):
        """
        """
        query_dict = dict()
        
        if not (('start date' in data_dict and 'end date' in data_dict) ^ ('date' in data_dict)):
            raise ValueError("Give as input either 'start date' and 'end date' or list of dates under key 'date'")
            
        if 'start date' in data_dict and 'end date' in data_dict:
            start = self._parse_dates(data_dict['start date'], sep)
            end = self._parse_dates(data_dict['end date'], sep)
            
            if len(start) > 1 or len(end) > 1:
                raise ValueError("Should provide single date as 'start date' or 'end date'")
                
            query_dict['date'] = [datetime.strftime(dt, "%Y-%m-%d") 
                                      for dt in pd.date_range(*start, *end)]
            
        for key in data_dict:
            if key not in ('start date', 'end date', 'weight'):
                value = self._parse_entry(key, data_dict[key], sep)
                query_dict[key] = value# handle error
                
        return query_dict
    
    
    def _parse_entry(self, key, data, sep):
        """
        """
        if key == 'geo':
            return self._parse_geos(data, sep)
        if key == 'vertical':
            return self._parse_verticals(data, sep)
        if key == 'logical_category':
            return self._parse_log_cats(data, sep)
        if key == 'channel':
            return self._parse_channels(data, sep)
        if key == 'date':
            return self._parse_dates(data, sep)
        else: 
            raise ValueError(f'Unknown key in input dict: {key}')
    
    
    def _parse_geos(self, data, sep):
        """
        """
        if isinstance(data, str):
            data = data.strip()
            data = data.split(sep)
        return data 
    
    
    def _parse_verticals(self, data, sep):
        """
        """
        if isinstance(data, str):
            data = data.strip()
            data = data.split(sep) 
        return data
        
    
    
    def _parse_log_cats(self, data, sep):
        """
        """
        if isinstance(data, str):
            data = data.strip()
            data = data.split(sep)
        return data
    
    
    def _parse_channels(self, data, sep):
        """
        """
        if isinstance(data, str):
            data = data.strip()
            data = data.split(sep)  
        return data
    
    
    def _parse_dates(self, data, sep):
        """
        """
        if isinstance(data, str):
            data = data.strip()
            data = data.split(sep)
        return data