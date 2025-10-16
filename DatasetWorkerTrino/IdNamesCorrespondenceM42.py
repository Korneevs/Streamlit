from typing import List
from bxtools.trino import TrinoEngine


class IdNamesCorrespondenceM42():

    @classmethod
    def regions_to_id(cls, regions: List[str], engine: TrinoEngine) -> List[str]:
        query = """
        select value_id AS id, value 
        from iceberg.ab.metrics_dimension_values 
        where dimension = 'region' 
        and is_active=true
        """
        reg_dict = engine.select(query).set_index('value')['id'].to_dict()
        reg_dict['Any'] = 'NULL'
        return [reg_dict[region] for region in regions]
    
    
    @classmethod
    def ids_to_regions(cls, region_ids: List[str], engine: TrinoEngine) -> List[str]:
        query = """
        select value_id AS id, value 
        from iceberg.ab.metrics_dimension_values 
        where dimension = 'region' 
        and is_active=true
        """
        reg_dict = engine.select(query).set_index('id')['value'].to_dict()
        return [reg_dict.get(reg_id, 'Undefined') for reg_id in region_ids]
    

    @classmethod
    def metrics_to_ids(cls, metrics: List[str], engine: TrinoEngine) -> List[str]:
        metric_cond = ', '.join([f"'{metric}'" for metric in metrics])
        query = f"""
        select title AS value, metric_id AS id
        from iceberg.ab.metrics_registry where title in ({metric_cond})
        """
        metric_dict = engine.select(query).set_index('value')['id'].to_dict()
        if len(set(metrics) - set(metric_dict)):
            raise ValueError("Not all ids found for metrics")
        return [metric_dict[metric] for metric in metrics]
    

    @classmethod
    def ids_to_metrics(cls, ids: List[str], engine: TrinoEngine) -> List[str]:
        ids_cond = ', '.join([str(id) for id in ids])
        query = f"""
        select title AS value, metric_id AS id
        from iceberg.ab.metrics_registry where metric_id in ({ids_cond})
        """
        metric_dict = engine.select(query).set_index('id')['value'].to_dict()
        if len(set(ids) - set(metric_dict)):
            raise ValueError("Not all metrics found for ids")
        return [metric_dict[id] for id in ids]
    

    @classmethod
    def verticals_to_id(cls, verticals: List[str], engine: TrinoEngine) -> List[str]:
        query = """
        select value_id AS id, value 
        from iceberg.ab.metrics_dimension_values 
        where dimension = 'vertical' 
        and is_active=true
        """
        vert_dict = engine.select(query).set_index('value')['id'].to_dict()
        vert_dict['Any'] = 'NULL'
        return [vert_dict[vert] for vert in verticals]


    @classmethod
    def ids_to_verticals(cls, vertical_ids: List[str], engine: TrinoEngine) -> List[str]:
        query = """
        select value_id AS id, value 
        from iceberg.ab.metrics_dimension_values 
        where dimension = 'vertical' 
        and is_active=true
        """
        vert_dict = engine.select(query).set_index('id')['value'].to_dict()
        return [vert_dict[vert_id] for vert_id in vertical_ids]
    
    @classmethod
    def logical_categories_to_id(cls, logical_category: List[str], engine: TrinoEngine) -> List[str]:
        query = """
        select value_id AS id, value 
        from iceberg.ab.metrics_dimension_values 
        where dimension = 'logical_category' 
        and is_active=true
        """
        lc_dict = engine.select(query).set_index('value')['id'].to_dict()
        lc_dict['Any'] = 'NULL'
        return [lc_dict[lc] for lc in logical_category]
    
    @classmethod
    def ids_to_logical_categories(cls, logical_category_ids: List[str], engine: TrinoEngine) -> List[str]:
        query = """
        select value_id AS id, value 
        from iceberg.ab.metrics_dimension_values 
        where dimension = 'logical_category' 
        and is_active=true
        """
        lc_dict = engine.select(query).set_index('id')['value'].to_dict()
        return [lc_dict[lc_id] for lc_id in logical_category_ids]
    
    @classmethod
    def ids_to_cities(cls, cities_ids: List[str], engine: TrinoEngine):
        query = """
        select value_id AS id, value 
        from iceberg.ab.metrics_dimension_values 
        where dimension = 'city' 
        and is_active=true
        """
        city_dict = engine.select(query).set_index('value')['id'].to_dict()
        return [city_dict.get(city_id, 'Undefined') for city_id in cities_ids]