from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from itertools import product
from media.ConfigHandlingTrino.SliceInfo import GeneralSliceInfo, MediaSliceInfo
from media.ConfigHandlingTrino.MetaInfo import MetaInfo
from media.ConfigHandlingTrino.RegionsRetrieval import RegionsRetrieval
from media.ConfigHandlingTrino.regions import regions as REGIONS
import pandas as pd

class SlicesRetrieval(ABC):
    """
    Extracts distinct slices from config 
    """

    @classmethod
    @abstractmethod
    def get_all_slices_from_config(cls, *args: Any, **kwargs: Any) -> List[Any]:
        """
        Creates list of slices from config
        """
        pass


    @classmethod
    @abstractmethod
    def get_all_slices_and_info_from_config(cls, *args: Any, **kwargs: Any) -> List[Any]:
        """
        Creates list of slices with corresponding important info from config
        """
        pass


class GeneralSlicesRetrieval(SlicesRetrieval):
    """
    Extracts distinct slices from configs (main_params and slices_json) for general purpose application 
    with media instruments without media-specific params
    """

    # Class dependencies
    regionsRetrieval = RegionsRetrieval
    generalSliceInfo = GeneralSliceInfo
    metaInfo = MetaInfo


    @classmethod
    def get_all_slices_from_config(cls, main_params: Dict[str, Any], slices_json: Dict[str, Any]) -> List[Any]:
        """
        Creates list of slices from media configs, consisting
        of main_params, slices_josn. Each slice is defined by name from slices_json
        and metric.

        Args:
            main_params
            slices_json 
        Returns:
            List[Any]: List of slice name and metric pairs
        """

        cls._check_main_params_fields(main_params)
        metrics = main_params['metrics']
        slice_names = slices_json.keys()
        assert len(slice_names)
        

        return list(product(slice_names, metrics))
    
    
    @classmethod
    def _create_ch_ids_dict(cls, engine):  
        regIds = engine.select("""
        select value_id, value 
        from iceberg.ab.metrics_dimension_values 
        where dimension = 'region' 
        and is_active=true""")
        reg_to_id = pd.Series(regIds.value_id.values, index=regIds.value).to_dict()
        reg_to_id['Any'] = 'NULL'

        logIds = engine.select("""
        select value_id, value 
        from iceberg.ab.metrics_dimension_values 
        where dimension = 'logical_category' 
        and is_active=true""")
        log_to_id = pd.Series(logIds.value_id.values, index=logIds.value).to_dict()
        log_to_id['Any'] = 'NULL'

        verIds = engine.select("""
        select value_id, value 
        from iceberg.ab.metrics_dimension_values 
        where dimension = 'vertical' 
        and is_active=true""")
        vert_to_id = pd.Series(verIds.value_id.values, index=verIds.value).to_dict()
        vert_to_id['Any'] = 'NULL'

        
    
        return reg_to_id, log_to_id, vert_to_id

    

    @classmethod
    def _make_link(cls, main_params, slice_dict, regions, reg_to_id, log_to_id, vert_to_id):
        reg_ids = ','.join([str(reg_to_id[r]) for r in regions])
        log_ids = ','.join([str(log_to_id[log]) for log in slice_dict['logical_category']])
        vertical_id = ','.join([str(vert_to_id[ver]) for ver in slice_dict['vertical']])
        link = (
            f"https://ab.k.avito.ru/metrics/m42/?"
            f"logical_category={log_ids}&"
            f"metric=1792&"
            f"region={reg_ids}&"
            f"report=main&"
            f"sum_by=logical_category,region&"
            f"vertical={vertical_id}"
        )
        return link
    

    @classmethod
    def get_all_slices_and_info_from_config(cls, 
                                            main_params: Dict[str, Any], 
                                            slices_json: Dict[str, Any],
                                            engine) -> List:
        """
        Creates list of slices with control/target regions (from main_params(!)) and dates
        from configs, consisting of main_params, slices_josn. 

        Args:
            main_params:
            slices_json:
            engine: Conection to engine (trino) for values validation
        Returns:
            List[Any]: List of slice name and metric pairs
        """

        slice_metric_pairs = cls.get_all_slices_from_config(main_params, slices_json)
        general_slices_list = []
        
        reg_to_id, log_to_id, vert_to_id = cls._create_ch_ids_dict(engine)
        
        for slice_name, metric in slice_metric_pairs:
            slice_dict = slices_json[slice_name]
            cls._check_slices_json_slice_fields(slice_dict)
            region_groups = cls.regionsRetrieval.make_regions(main_params)
            
            if 'target' not in slice_dict:
                slice_dict['target'] = cls._make_link(main_params, slice_dict, region_groups['test regions'],
                                                      reg_to_id, log_to_id, vert_to_id)
            
            general_slice = cls.generalSliceInfo(
                            slice_name=slice_name,
                            metric=metric,
                            test_regions=list(region_groups['test regions']),
                            control_regions=list(region_groups['control regions']),
                            exclude_regions=list(region_groups['exclude regions']),
                            start_date=main_params['analysed_start_date'],
                            end_date=main_params['analysed_end_date'],
                            data_source_string=slice_dict['target'],
                            features=None)
            general_slices_list.append(general_slice)
        return general_slices_list
    
    
    @classmethod
    def get_meta_info(cls, main_params) -> MetaInfo:
        cls._check_main_params_fields(main_params)
        meta_data = MetaInfo(metrics=main_params['metrics'],
                             start_date=main_params['analysed_start_date'],
                             end_date=main_params['analysed_end_date'],
                             flight_end_date=main_params['flight_end_date'],
                             label=main_params['label'] if 'label' in main_params else main_params['flight_name'])
        return meta_data
            
    
    @classmethod
    def _check_main_params_fields(cls, main_params):
        assert 'metrics' in main_params
        assert isinstance(main_params['metrics'], list), "Метрики должны быть списком"
        assert len(main_params['metrics'])
        assert 'test regions' in main_params
        assert 'analysed_start_date' in main_params
        assert 'analysed_end_date' in main_params
        assert 'control regions' in main_params
        assert 'exclude regions' in main_params

    
    @classmethod            
    def _check_slices_json_slice_fields(cls, slice_dict):
        pass
#         assert 'target' in slice_dict



class MediaSlicesRetrieval(GeneralSlicesRetrieval):
    """
    Extracts distinct slices from media configs (main_params and slices_json) with media-specific
    params for specifically analyising media
    """

    # Class dependencies
    regionsRetrieval = RegionsRetrieval
    mediaSliceInfo = MediaSliceInfo
    metaInfo = MetaInfo
    
    @classmethod
    def _create_ch_ids_dict(cls, engine):
        regIds = engine.select("""
            select value_id, value 
            from iceberg.ab.metrics_dimension_values 
            where dimension = 'region' 
            and is_active=true""")
        reg_to_id = pd.Series(regIds.value_id.values, index=regIds.value).to_dict()
        reg_to_id['Any'] = 'NULL'

        logIds = engine.select("""
            select value_id, value 
            from iceberg.ab.metrics_dimension_values 
            where dimension = 'logical_category' 
            and is_active=true""")
        log_to_id = pd.Series(logIds.value_id.values, index=logIds.value).to_dict()
        log_to_id['Any'] = 'NULL'

        verIds = engine.select("""
            select value_id, value 
            from iceberg.ab.metrics_dimension_values 
            where dimension = 'vertical' 
            and is_active=true""")
        vert_to_id = pd.Series(verIds.value_id.values, index=verIds.value).to_dict()
        vert_to_id['Any'] = 'NULL'

        return reg_to_id, log_to_id, vert_to_id

    

    @classmethod
    def _make_link(cls, main_params, slice_dict, regions, reg_to_id, log_to_id, vert_to_id):
        reg_ids = ','.join([str(reg_to_id[r]) for r in regions])
        log_ids = ','.join([str(log_to_id[log]) for log in slice_dict['logical_category']])
        vertical_id = ','.join([str(vert_to_id[ver]) for ver in slice_dict['vertical']])
        link = (
            f"https://ab.k.avito.ru/metrics/m42/?"
            f"logical_category={log_ids}&"
            f"metric=1792&"
            f"region={reg_ids}&"
            f"report=main&"
            f"sum_by=logical_category,region&"
            f"vertical={vertical_id}"
        )
        return link
    
    
    @classmethod
    def get_all_slices_and_info_from_config_with_dtb(cls, 
                                            main_params: Dict[str, Any], 
                                            slices_json: Dict[str, Any],
                                            engine) -> List[Any]:
        """
        Creates List of slices with vertical, logacts, target/control regions (from main_params(!)),
        dates and budgets from media configs, consisting of main_params, slices_josn

        Args:
            main_params:
            slices_json:
            engine: Conection to engine for values validation
        Returns:
            List[Any]: List of slice name and metric pairs
        """

        slice_metric_pairs = cls.get_all_slices_from_config(main_params, slices_json)
        media_slices_list = []
        media_slice_keys = set()
        
        reg_to_id, log_to_id, vert_to_id = cls._create_ch_ids_dict(engine)
        for slice_name, metric in slice_metric_pairs:
            slice_dict = slices_json[slice_name]
            cls._check_slices_json_slice_fields(slice_dict)
            region_groups = cls.regionsRetrieval.make_regions(main_params)
            
            if 'target' not in slice_dict:
                slice_dict['target'] = cls._make_link(main_params, slice_dict, region_groups['test regions'],
                                                      reg_to_id, log_to_id, vert_to_id)
                
            media_slice = cls.mediaSliceInfo(
                            slice_name=slice_name,
                            metric=metric,
                            vertical=slice_dict['vertical'],
                            logical_category=slice_dict['logical_category'],
                            test_regions=list(region_groups['test regions']),
                            control_regions=list(region_groups['control regions']),
                            exclude_regions=list(region_groups['exclude regions']),
                            start_date=main_params['analysed_start_date'],
                            end_date=main_params['analysed_end_date'],
                            budget=main_params['flight_budget'] if 'flight_budget' in main_params
                                        else slice_dict['analysed_budget'],
                            data_source_string=slice_dict['target'],
                            features=None)
            media_slices_list.append(media_slice)
            
            media_slice_key = media_slice.media_slice_key
            if media_slice_key not in media_slice_keys:
                media_slice_keys.add(media_slice_key)
                corresponding_dtb_slice = media_slice.get_corresponding_dtb_slice(engine)
                media_slices_list.append(corresponding_dtb_slice)

        return media_slices_list
    
    
    @classmethod
    def get_all_slices_and_info_from_config(cls, 
                                            main_params: Dict[str, Any], 
                                            slices_json: Dict[str, Any]) -> List[Any]:
        """
        Creates List of slices with vertical, logacts, target/control regions (from main_params(!)),
        dates and budgets from media configs, consisting of main_params, slices_josn

        Args:
            main_params:
            slices_json:
            engine: Conection to engine (trino) for values validation
        Returns:
            List[Any]: List of slice name and metric pairs
        """

        slice_metric_pairs = cls.get_all_slices_from_config(main_params, slices_json)
        media_slices_list = []

        for slice_name, metric in slice_metric_pairs:
            slice_dict = slices_json[slice_name]
            cls._check_slices_json_slice_fields(slice_dict)
            region_groups = cls.regionsRetrieval.make_regions(main_params)

            media_slice = cls.mediaSliceInfo(
                            slice_name=slice_name,
                            metric=metric,
                            vertical=slice_dict['vertical'],
                            logical_category=slice_dict['logical_category'],
                            test_regions=list(region_groups['test regions']),
                            control_regions=list(region_groups['control regions']),
                            exclude_regions=list(region_groups['exclude regions']),
                            start_date=main_params['analysed_start_date'],
                            end_date=main_params['analysed_end_date'],
                            budget=main_params['flight_budget'] if 'flight_budget' in main_params
                                        else slice_dict['analysed_budget'],
                            data_source_string=slice_dict['target'] if 'target' in slice_dict else None,
                            features=None)
            media_slices_list.append(media_slice)

        return media_slices_list

    
    @classmethod            
    def _check_slices_json_slice_fields(cls, slice_dict):
        assert 'vertical' in slice_dict
        assert 'logical_category' in slice_dict
#         assert 'analysed_budget' in slice_dict
#         assert 'target' in slice_dict    