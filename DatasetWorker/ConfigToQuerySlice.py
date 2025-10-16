from media.ConfigHandling.SliceInfo import SliceInfo, GeneralSliceInfo, MediaSliceInfo
from media.DatasetWorker.QuerySliceInfo import QuerySliceInfo, MediaQuerySliceInfo
from collections import defaultdict
from typing import List, Any


class ConfigToQuerySlice():

    @classmethod
    def get_query_slices_from_config_slices(cls, config_slices: List[GeneralSliceInfo]) -> List[QuerySliceInfo]:
        # One query for each slice_name and all metrics
        slice_name_metrics = defaultdict(list)
        slice_name_slice_info = dict()
        for slice_with_info in config_slices:
            slice_name = slice_with_info.slice_name
            slice_name_metrics[slice_name].append(slice_with_info.metric)
            if slice_name not in slice_name_slice_info:
                slice_name_slice_info[slice_name] = cls.get_query_slice_from_config_slice(slice_with_info)
        
        for slice_name, metrics_list in slice_name_metrics.items():
            slice_name_slice_info[slice_name].metrics_list = metrics_list
        
        return list(slice_name_slice_info.values())
    
    
    @classmethod
    def get_query_slice_from_config_slice(cls, slice_with_info: GeneralSliceInfo) -> QuerySliceInfo:
        QuerySlice = QuerySliceInfo(
                    slice_name=slice_with_info.slice_name,
                    metrics_list=[slice_with_info.metric],
                    test_regions=slice_with_info.test_regions,
                    control_regions=slice_with_info.control_regions,
                    exclude_regions=slice_with_info.exclude_regions,
                    start_date=slice_with_info.start_date,
                    end_date=slice_with_info.end_date,
                    data_source_string=slice_with_info.data_source_string,
                    data_source_type=slice_with_info.data_source_type,
                    features=slice_with_info.features
                )
        return QuerySlice
    
    
class MediaConfigToQuerySlice():

    @classmethod
    def get_query_slices_from_config_slices(cls, config_slices: List[MediaSliceInfo]) -> List[MediaQuerySliceInfo]:
        # One query for each slice_name and all metrics
        slice_name_metrics = defaultdict(list)
        slice_name_slice_info = dict()
        for slice_with_info in config_slices:
            slice_name = slice_with_info.slice_name
            slice_name_metrics[slice_name].append(slice_with_info.metric)
            if slice_name not in slice_name_slice_info:
                slice_name_slice_info[slice_name] = cls.get_query_slice_from_config_slice(slice_with_info)
        
        for slice_name, metrics_list in slice_name_metrics.items():
            slice_name_slice_info[slice_name].metrics_list = metrics_list
        
        return list(slice_name_slice_info.values())
    
    
    @classmethod
    def get_query_slice_from_config_slice(cls, slice_with_info: MediaSliceInfo) -> MediaQuerySliceInfo:
        mediaQuerySlice = MediaQuerySliceInfo(
                    slice_name=slice_with_info.slice_name,
                    metrics_list=[slice_with_info.metric],
                    test_regions=slice_with_info.test_regions,
                    control_regions=slice_with_info.control_regions,
                    exclude_regions=slice_with_info.exclude_regions,
                    start_date=slice_with_info.start_date,
                    end_date=slice_with_info.end_date,
                    data_source_string=slice_with_info.data_source_string,
                    data_source_type=slice_with_info.data_source_type,
                    features=slice_with_info.features,
                    vertical=slice_with_info.vertical,
                    logical_category=slice_with_info.logical_category,
                    budget=slice_with_info.budget
                )
        return mediaQuerySlice
        
        
    
    
    