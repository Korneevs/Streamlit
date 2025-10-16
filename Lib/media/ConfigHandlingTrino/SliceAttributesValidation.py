from bxtools.trino import TrinoEngine
from typing import List
import datetime


class SliceInfoError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class SliceAttributesValidation():
    @classmethod
    def check_vertical(cls, vertical: List[str], engine: TrinoEngine) -> None:
        try: 
            query = """
                select value 
                from iceberg.ab.metrics_dimension_values 
                where dimension = 'vertical' 
                and is_active=true
                """
            good_values = set(engine.select(query)['value'])
        except:
            raise SliceInfoError('Unsuccesful query to trino iceberg.ab.metrics_dimension_values for \
                                 config vertical parameter validation')
        if len(set(vertical) - set(good_values)) > 0:
            bad = set(vertical) - set(good_values)
            raise SliceInfoError(f"Bad vertical names: {bad}")


    @classmethod
    def check_logical_category(cls, logical_category: List[str], engine: TrinoEngine) -> None:
        try: 
            query = """
                select value 
                from iceberg.ab.metrics_dimension_values 
                where dimension = 'logical_category' 
                and is_active=true
                """
            good_values = set(engine.select(query)['value'])
        except:
            raise SliceInfoError('Unsuccesful query to trino iceberg.ab.metrics_dimension_values for \
                                 config logical_category parameter validation')
        if len(set(logical_category) - set(good_values)) > 0:
            bad = set(logical_category) - set(good_values)
            raise SliceInfoError(f"Bad logical_category names: {bad}")
        

    @classmethod
    def check_m42_metric_name(cls, metric: str, engine: TrinoEngine) -> None:
        query = f"""
        select title AS value
        from iceberg.ab.metrics_registry where title in ('{metric}')
        """
        metric_df = engine.select(query)
        if len(metric_df) == 0:
            raise SliceInfoError(f"""В m42 не нашлась метрика {metric}. Проверьте наличие метрики на сайте https://ab.k.avito.ru/metrics/m42/""")


    @classmethod
    def check_m42_metric_name_and_is_not_ratio(cls, metric: str, engine: TrinoEngine) -> None:
        query = f"""
        select title AS value, type
        from iceberg.ab.metrics_registry where title in ('{metric}')
        """
        metric_df = engine.select(query)
        if len(metric_df) == 0:
            raise SliceInfoError(f"""В m42 не нашлась метрика {metric}. Проверьте наличие метрики на сайте https://ab.k.avito.ru/metrics/m42/""")
        bad_df = metric_df[metric_df['type'] == 'ratio']
        if len(bad_df):
            bad_metrics = list(bad_df['metrics'])
            raise SliceInfoError(f"""Нет поддержки ratio метрик, а точнее {bad_metrics}.""")
        

    @classmethod
    def check_dates(cls, start_date: str, end_date: str) -> None:
        assert datetime.datetime.strptime(end_date, '%Y-%m-%d') >= datetime.datetime.strptime(start_date, '%Y-%m-%d')