from media.DatasetWorker.IdNamesCorrespondenceM42 import IdNamesCorrespondenceM42
from bxtools.clickhouse import CHEngine
from typing import List


class LinkGeneratorM42():

    #class dependencies
    idNamesCorrespondenceM42 = IdNamesCorrespondenceM42

    @classmethod
    def get_link(cls, vertical: List[str], logical_category: List[str], 
                 target_regions: List[str], metrics: List[str], ch_engine: CHEngine):
        
        reg_ids = cls.idNamesCorrespondenceM42.regions_to_id(target_regions, ch_engine)
        reg_ids = ','.join([str(id) for id in reg_ids])
        log_ids = cls.idNamesCorrespondenceM42.logical_categories_to_id(logical_category, ch_engine)
        log_ids = ','.join([str(id) for id in log_ids])
        vertical_ids = cls.idNamesCorrespondenceM42.verticals_to_id(vertical, ch_engine)
        vertical_ids = ','.join([str(id) for id in vertical_ids])
        metric_ids = cls.idNamesCorrespondenceM42.metrics_to_ids(metrics, ch_engine)
        metric_ids = ','.join([str(id) for id in metric_ids])

        link = f"https://m42.k.avito.ru/?&logical_category={log_ids}&metric={metric_ids}\
&region={reg_ids}&report=main&sum_by=vertical,logical_category,region&vertical={vertical_ids}"
        return link