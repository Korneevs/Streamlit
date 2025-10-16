from media.ConfigHandling.SlicesRetrieval import MediaSlicesRetrieval
from media.LongTermEffects.lt_params_config import LT
from typing import List


class LongTermEffect():

    def __init__(self):

        #Class dependencies
        self.mediaSlicesRetrieval = MediaSlicesRetrieval


    def get_lt_coef(self, main_params, slices_json):

        lt_dict = {}

        slices_with_info = self.mediaSlicesRetrieval.get_all_slices_and_info_from_config(
            main_params, 
            slices_json)
        
        for slice_info in slices_with_info:
            if slice_info.is_dtb_slice:
                continue
            lt_dict[(slice_info.slice_name, slice_info.metric)] = LT

        return lt_dict