from bxtools.clickhouse import CHEngine
from typing import List
from datetime import datetime


class SliceInfoError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class SliceAttributesValidation():
    @classmethod
    def check_vertical(cls, vertical: List[str], ch_engine: CHEngine) -> None:
        try: 
            good_values = set(ch_engine.select(f'SELECT value from dct.m42_vertical')['value'])
        except:
            raise SliceInfoError('Unsuccesful query to clickhouse dct.m42_vertical for \
                                 config parameter validation')
        if len(set(vertical) - set(good_values)) > 0:
            bad = set(vertical) - set(good_values)
            raise SliceInfoError(f"Bad vertical names: {bad}")


    @classmethod
    def check_logical_category(cls, logical_category: List[str], ch_engine: CHEngine) -> None:
        try: 
            good_values = set(ch_engine.select(f'SELECT value from dct.m42_logical_category')['value'])
        except:
            raise SliceInfoError('Unsuccesful query to clickhouse dct.m42_logical_category for \
                                 config parameter validation')
        if len(set(logical_category) - set(good_values)) > 0:
            bad = set(logical_category) - set(good_values)
            raise SliceInfoError(f"Bad logical_category names: {bad}")
        

    @classmethod
    def check_m42_metric_name(cls, metric: str, ch_enigne: CHEngine) -> None:
        metric_df = ch_enigne.select(f"SELECT * FROM dct.m42_metric WHERE metric in ('{metric}')")
        if len(metric_df) == 0:
            raise SliceInfoError(f"""В m42 не нашлась метрика {metric}. Проверьте наличие метрики на сайте https://m42.k.avito.ru/""")


    @classmethod
    def check_m42_metric_name_and_is_not_ratio(cls, metric: str, ch_enigne: CHEngine) -> None:
        metric_df = ch_enigne.select(f"SELECT * FROM dct.m42_metric WHERE metric in ('{metric}')")
        if len(metric_df) == 0:
            raise SliceInfoError(f"""В m42 не нашлась метрика {metric}. Проверьте наличие метрики на сайте https://m42.k.avito.ru/""")
        bad_df = metric_df[metric_df['type'] == 'ratio']
        if len(bad_df):
            bad_metrics = list(bad_df['metrics'])
            raise SliceInfoError(f"""Нет поддержки ratio метрик, а точнее {bad_metrics}.""")
        

    @classmethod
    def check_dates(cls, start_date: str, end_date: str) -> None:
        assert datetime.strptime(end_date, '%Y-%m-%d') >= datetime.strptime(start_date, '%Y-%m-%d')