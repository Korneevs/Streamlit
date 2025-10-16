from typing import List
from bxtools.clickhouse import CHEngine


class IdNamesCorrespondenceM42():

    @classmethod
    def regions_to_id(cls, regions: List[str], ch_engine: CHEngine) -> List[str]:
        reg_dict = ch_engine.select(f"SELECT * FROM dct.m42_region").set_index('value')['id'].to_dict()
        reg_dict['Any'] = -1
        return [reg_dict[region] for region in regions]
    
    
    @classmethod
    def ids_to_regions(cls, region_ids: List[str], ch_engine: CHEngine) -> List[str]:
        reg_dict = ch_engine.select(f"SELECT * FROM dct.m42_region").set_index('id')['value'].to_dict()
        return [reg_dict[reg_id] for reg_id in region_ids]
    

    @classmethod
    def metrics_to_ids(cls, metrics: List[str], ch_engine: CHEngine) -> List[str]:
        metric_cond = ', '.join([f"'{metric}'" for metric in metrics])
        metric_dict = ch_engine.select(f"SELECT * FROM dct.m42_metric WHERE metric in ({metric_cond})").set_index('value')['id'].to_dict()
        if len(set(metrics) - set(metric_dict)):
            raise ValueError("Not all ids found for metrics")
        return [metric_dict[metric] for metric in metrics]
    

    @classmethod
    def ids_to_metrics(cls, ids: List[str], ch_engine: CHEngine) -> List[str]:
        ids_cond = ', '.join([str(id) for id in ids])
        metric_dict = ch_engine.select(f"SELECT * FROM dct.m42_metric WHERE id in ('{ids_cond}')").set_index('id')['value'].to_dict()
        if len(set(ids) - set(metric_dict)):
            raise ValueError("Not all metrics found for ids")
        return [metric_dict[id] for id in ids]
    

    @classmethod
    def verticals_to_id(cls, verticals: List[str], ch_engine: CHEngine) -> List[str]:
        vert_dict = ch_engine.select(f"SELECT * FROM dct.m42_vertical").set_index('value')['id'].to_dict()
        vert_dict['Any'] = -1
        return [vert_dict[vert] for vert in verticals]


    @classmethod
    def ids_to_verticals(cls, vertical_ids: List[str], ch_engine: CHEngine) -> List[str]:
        vert_dict = ch_engine.select(f"SELECT * FROM dct.m42_vertical").set_index('id')['value'].to_dict()
        return [vert_dict[vert_id] for vert_id in vertical_ids]
    
    @classmethod
    def logical_categories_to_id(cls, logical_category: List[str], ch_engine: CHEngine) -> List[str]:
        lc_dict = ch_engine.select(f"SELECT * FROM dct.m42_logical_category").set_index('value')['id'].to_dict()
        lc_dict['Any'] = -1
        return [lc_dict[lc] for lc in logical_category]
    
    @classmethod
    def ids_to_logical_categories(cls, logical_category_ids: List[str], ch_engine: CHEngine) -> List[str]:
        lc_dict = ch_engine.select(f"SELECT * FROM dct.m42_logical_category").set_index('id')['value'].to_dict()
        return [lc_dict[lc_id] for lc_id in logical_category_ids]