from media.CrossEffects.CrossEffect import CrossEffect
from media.ConfigHandling.SlicesRetrieval import MediaSlicesRetrieval
from typing import Dict, Any, Tuple


class CrossEffectsForSlices():

    def __init__(self, engine):
        
        self.engine = engine

        #Class dependencies
        self.crossEfffect = CrossEffect(engine)
        self.mediaSlicesRetrieval = MediaSlicesRetrieval


    def get_crosseffects_for_slices(self, main_params, slices_json) -> dict:

        slices_ce = {}

        ce_data_dict = self.crossEfffect.get_data_for_crosseffect()

        slices_with_info = self.mediaSlicesRetrieval.get_all_slices_and_info_from_config(
            main_params, 
            slices_json)
        
        for slice_info in slices_with_info:
            if slice_info.is_dtb_slice:
                continue
            target_cats = slice_info.logical_category
            ce = self.crossEfffect.get_crosseffect_with_calculated_data(ce_data_dict, target_cats)
            slices_ce[(slice_info.slice_name, slice_info.metric)] = ce
        
        return slices_ce


    