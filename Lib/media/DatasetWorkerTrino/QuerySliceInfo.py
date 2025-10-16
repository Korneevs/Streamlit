from dataclasses import dataclass, field
from typing import List, Any, Optional
from media.DatasetWorkerTrino.LinkGeneratorM42 import LinkGeneratorM42


@dataclass
class QuerySliceInfo():
    slice_name: str
    metrics_list: List[str]
    test_regions: List[str]
    control_regions: List[str]
    exclude_regions: List[str]
    start_date: str
    end_date: str
    data_source_string: str
    data_source_type: str
    features: Optional[List[Any]] # Not used
        
        
@dataclass
class MediaQuerySliceInfo(QuerySliceInfo):
    vertical: List[str]
    logical_category: List[str]
    budget: float

        