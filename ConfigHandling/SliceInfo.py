from bxtools.clickhouse import CHEngine
from dataclasses import dataclass, field
from abc import ABC
from typing import List, Optional, Any, Tuple
from media.ConfigHandling.SliceAttributesValidation import SliceAttributesValidation
from media.DatasetWorker.LinkGeneratorM42 import LinkGeneratorM42



@dataclass
class SliceInfo(ABC):
    pass


@dataclass
class GeneralSliceInfo(SliceInfo):
    slice_name: str
    metric: str
    test_regions: List[str]
    control_regions: List[str] 
    exclude_regions: List[str]
    start_date: str
    end_date: str
    features: Optional[List[Any]] # Not used 
    data_source_string: str
    data_source_type: str = field(init=False) 


    def __post_init__(self):
        self.data_source_type = self.get_data_source_type(self.data_source_string)

        #Class Dependencies
        self.sliceAttributesValidation = SliceAttributesValidation



    def validate_attributes(self, ch_engine: Optional[CHEngine] = None):
        if self.data_source_type == 'm42':
            assert isinstance(ch_engine, CHEngine)
            self.sliceAttributesValidation.check_m42_metric_name_and_is_not_ratio(self.metric, ch_engine)
        self.sliceAttributesValidation.check_dates(self.start_date, self.end_date)


    def get_data_source_type(self, data_source_str):
        if '//m42.k.avito.ru/' in data_source_str:
            return 'm42'
        if data_source_str[-4:] == '.sql':
            return 'sql file'
        if data_source_str[-4:] == '.csv':
            return 'csv file'
        return 'unknown'


@dataclass
class MediaSliceInfo(GeneralSliceInfo):
    vertical: List[str]
    logical_category: List[str]
    budget: str
    # hashable key - uniquely depends on vertical, logcats, dates and regions 
    # (used to find corresponding DTB slices)
    media_slice_key: Tuple[Any] = field(init=False) 
    is_dtb_slice: bool = False


    def __post_init__(self):

        #Class Dependencies
        self.sliceAttributesValidation = SliceAttributesValidation
        self.linkGeneratorM42 = LinkGeneratorM42

        self.data_source_type = self.get_data_source_type(self.data_source_string)
        self.media_slice_key = self.get_media_slice_key()


    def validate_attributes(self, ch_engine: CHEngine):
        if self.data_source_type == 'm42':
            assert isinstance(ch_engine, CHEngine)
            self.sliceAttributesValidation.check_m42_metric_name_and_is_not_ratio(self.metric, ch_engine)
        self.sliceAttributesValidation.check_dates(self.start_date, self.end_date)
        self.sliceAttributesValidation.check_vertical(self.vertical, ch_engine)
        self.sliceAttributesValidation.check_logical_category(self.logical_category, ch_engine)


    def get_media_slice_key(self) -> Tuple[Any]:
        vertical = tuple(sorted(list(set(self.vertical))))
        logical_category = tuple(sorted(list(set(self.logical_category))))
        test_regions = tuple(sorted(list(set(self.test_regions))))
        control_regions = tuple(sorted(list(set(self.control_regions))))
        exclude_regions = tuple(sorted(list(set(self.exclude_regions))))
        start_date = self.start_date
        end_date = self.end_date
        return (vertical, logical_category, test_regions, control_regions, 
              exclude_regions, start_date, end_date)
    
    
    def get_corresponding_dtb_slice(self, ch_engine: CHEngine) -> 'MediaSliceInfo':
        metric = 'DTB'
        dtb_data_source = self.linkGeneratorM42.get_link(self.vertical, self.logical_category, 
                                                            self.test_regions, [metric], ch_engine)
        return MediaSliceInfo(
            slice_name=f'{self.slice_name}',
            metric=metric,
            vertical=self.vertical,
            logical_category=self.logical_category,
            test_regions=self.test_regions,
            control_regions=self.control_regions,
            exclude_regions=self.exclude_regions,
            start_date=self.start_date,
            end_date=self.end_date,
            budget=self.budget,
            data_source_string=dtb_data_source,
            is_dtb_slice=True,
            features=None)
    
    
    def get_corresponding_federal_dtb_slice(self, ch_engine: CHEngine) -> 'MediaSliceInfo':
        metric = 'DTB'
        dtb_data_source = self.linkGeneratorM42.get_link(self.vertical, self.logical_category, 
                                                            ['Any'], [metric], ch_engine)
        return MediaSliceInfo(
            slice_name=f'{self.slice_name}',
            metric=metric,
            vertical=self.vertical,
            logical_category=self.logical_category,
            test_regions=['Any'],
            control_regions=[],
            exclude_regions=[],
            start_date=self.start_date,
            end_date=self.end_date,
            budget=self.budget,
            data_source_string=dtb_data_source,
            is_dtb_slice=True,
            features=None)


