from abc import ABC
from media.media_analyser.EffectRetrieval import EffectRetrieval
from media.ConfigHandling.SlicesRetrieval import MediaSlicesRetrieval
from media.DatasetWorker.ConfigToQuerySlice import MediaConfigToQuerySlice
from scipy.stats import norm
from numpy.random import normal
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
from dateutil.relativedelta import relativedelta


class MediaEffectsArrangement():
    
    #Class dependencies
    effectRetrieval = EffectRetrieval
    mediaSlicesRetrieval = MediaSlicesRetrieval
    mediaConfigToQuerySlice = MediaConfigToQuerySlice


    @classmethod
    def get_effect_components(cls, main_params, slices_json,
                              ts_df, metric_value_dict, ce_dict, 
                              lt_dict, dc_results, dc_creator, main_metric) -> Dict[str, Any]:
        """
        Get ROMIs from results of regAB/YoY, metric value calculations, ce calculations, lt calculations, dtb over year, etc.
        """
        
        ch_engine = dc_creator.ch_engine
        assert ch_engine is not None
        
        template_dicts = []
        effects_df = cls.effectRetrieval.get_effects(ts_df)
        slices_with_info = cls.mediaSlicesRetrieval.get_all_slices_and_info_from_config_with_dtb(
            main_params, slices_json, ch_engine)
        
        dtb_slices = {dtb_slice.media_slice_key: dtb_slice for dtb_slice in slices_with_info if dtb_slice.is_dtb_slice}
        assert len(dtb_slices), 'Данных по DTB нет в dc_result, проверьте, что при загрузке в DatasetCreator указано for_media=True'

        
        for slice_with_info in slices_with_info:
            
            if slice_with_info.is_dtb_slice:
                continue # DTB is not target slice, only supplements report for target slice
                
            slice_name, metric = slice_with_info.slice_name, slice_with_info.metric
            if metric != main_metric:
                continue
#             print(metric)
            budget = slice_with_info.budget
            
            if (slice_name, metric) in metric_value_dict:
                total_elasticity = metric_value_dict[(slice_name, metric)]['total_elasticity']
                total_elasticity_std = metric_value_dict[(slice_name, metric)]['total_elasticity_std']
            else:
                continue # TODO: Make normal slices with dataloader and regAB and remove this
                
            row = effects_df.query("slice == @slice_name and metric == @metric")
            if not len(row):
                continue # TODO: Make normal slices with dataloader and regAB 
                
                
            #Get target metric data
            metric_data = dc_results['ml_datasets'][(metric, slice_name)]['dataset']
            metric_over_campaign = cls.get_metric_over_campaign(metric_data=metric_data,
                                                                test_regions=slice_with_info.test_regions,
                                                                start_date=slice_with_info.start_date,
                                                                end_date=slice_with_info.end_date)
            
            
            # Get target metric effect
            # Asssuming that relative effect and total elasticity is normal!
            alpha = row["alpha"].iloc[0]
            
            
            rel_effect = row["rel. effect"].iloc[0]
            rel_sigma = row['rel. sigma'].iloc[0]
            rel_effect_left = row['rel. effect lower bound'].iloc[0]
            rel_effect_right = row['rel. effect upper bound'].iloc[0]
            
            
            abs_promo_effect = row['abs. effect'].iloc[0]
            abs_sigma = row['abs. sigma']
            abs_promo_effect_left = row['abs. effect lower bound'].iloc[0]
            abs_promo_effect_right = row['abs. effect upper bound'].iloc[0]
            
            # Get elasticity wrt liquidity metrics data and total revenue
            z = norm.ppf(1 - alpha)
            total_elasticity_left, total_elasticity_right = total_elasticity - z * total_elasticity_std, \
                total_elasticity + z * total_elasticity_std
            total_revenue = metric_value_dict[(slice_name, metric)]['total_revenue']
            
            # Get target metric value in rubles (w/o elasticity correction)
            metric_weight = total_revenue / metric_over_campaign
            metric_value = total_elasticity * metric_weight
            metric_value_left = total_elasticity_left * total_revenue / metric_over_campaign
            metric_value_right = total_elasticity_right * total_revenue / metric_over_campaign
            
            # Get st promo estimates
            st_promo_romi_left, st_promo_romi, st_promo_romi_right = cls.get_romi_from_elasticity_uplift(
                rel_effect, rel_sigma, alpha, budget, total_elasticity, total_elasticity_std, total_revenue)
            st_promo_gross_roi_left, st_promo_gross_roi, st_promo_gross_roi_right = st_promo_romi_left + 1, \
                st_promo_romi + 1, st_promo_romi_right + 1
            
            st_promo_romi_from_abs_left, st_promo_romi_from_abs, st_promo_romi_from_abs_right = cls.get_romi_from_elasticity_uplift_and_abs_effect(abs_promo_effect, abs_sigma, alpha, budget, total_elasticity, total_elasticity_std, metric_weight)
            
            st_promo_gross_roi_from_abs = st_promo_romi_from_abs + 1
            st_promo_gross_roi_from_abs_left = st_promo_romi_from_abs_left + 1
            st_promo_gross_roi_from_abs_right = st_promo_romi_from_abs_right + 1
            
            
            # Get st with ce estimates
            if (slice_name, metric) in ce_dict:
                ce = ce_dict[(slice_name, metric)]
            else:
                continue # TODO: Make normal slices with dataloader and regAB 

            st_gross_roi_left, st_gross_roi, st_gross_roi_right = st_promo_gross_roi_left * ce, \
                st_promo_gross_roi * ce, st_promo_gross_roi_right * ce 
            
            # Get lt estimates
            if (slice_name, metric) in lt_dict:
                lt = lt_dict[(slice_name, metric)]
            else:
                continue # TODO: Make normal slices with dataloader and regAB 

            gross_roi_left, gross_roi, gross_roi_right = st_gross_roi_left * lt, st_gross_roi * lt, \
                st_gross_roi_right * lt
            
            st_romi_left, st_romi, st_romi_right = st_gross_roi_left - 1, st_gross_roi - 1, \
                st_gross_roi_right - 1

            romi_left, romi, romi_right = gross_roi_left - 1, gross_roi - 1, gross_roi_right - 1
            
            abs_lt_effect = abs_promo_effect * lt
            abs_lt_effect_left, abs_lt_effect_right = abs_promo_effect_left * lt, abs_promo_effect_right * lt
            
            romi_from_abs_left = (st_promo_romi_from_abs_left + 1) * ce * lt - 1
            romi_from_abs = (st_promo_romi_from_abs + 1) * ce * lt - 1 
            romi_from_abs_right = (st_promo_romi_from_abs_right + 1) * ce * lt -1
            
            rel_effect_lt = rel_effect * lt
            rel_effect_lt_left, rel_effect_lt_right = rel_effect_left * lt, rel_effect_right * lt
            
            # Get corresponding DTB data
            dtb_slice = dtb_slices[slice_with_info.media_slice_key] # corresponding dtb slice
            
            dtb_slice_name = dtb_slice.slice_name
#             if dtb_slice_name not in effects_df['slice'].unique():
            dtb_slice_name = dtb_slice_name.split(' : ')[0]
            dtb_metric_name = dtb_slice.metric
            
#             dtb_row = effects_df.query("slice == @dtb_slice_name and metric == @dtb_metric_name")
#             print(metric, dtb_slice_name)
            dtb_row = effects_df.query("slice == @dtb_slice_name and metric == @metric")
            if len(dtb_row):
                dtb_rel = dtb_row['rel. effect'].iloc[0]
                dtb_federal_slice = dtb_slice.get_corresponding_federal_dtb_slice(ch_engine)
                dtb_query_slice = cls.mediaConfigToQuerySlice.get_query_slice_from_config_slice(dtb_federal_slice)
#                 print(dtb_query_slice)

                dtb_fed_data = dc_creator.get_one_slice_dataset(dtb_query_slice)
                dtb_fed_df = dtb_fed_data[('DTB', dtb_query_slice.slice_name)]['dataset']
                dtb_df = dc_results['ml_datasets'][(dtb_metric_name, dtb_slice_name)]['dataset']
                
#                 print(dtb_fed_df)
                dtb_over_year = cls.get_dtb_over_year(dtb_data=dtb_fed_df, test_regions=dtb_federal_slice.test_regions,
                                      end_date=dtb_federal_slice.end_date)

                dtb_over_campaign = cls.get_dtb_over_campaign(dtb_data=dtb_df, test_regions=dtb_slice.test_regions,
                                      start_date=dtb_slice.start_date, end_date=dtb_slice.end_date)

                
                dtb_rel_effect_over_year = dtb_rel * dtb_over_campaign / dtb_over_year
            
                dtb_rel_effect_over_year_with_lt = dtb_rel_effect_over_year * lt
            else: 
                dtb_over_year = 0
                dtb_over_campaign = 0
                dtb_rel = 0
                dtb_rel_effect_over_year = 0
                dtb_rel_effect_over_year_with_lt = 0
                
            
            # Get verdict
            campaign_result = 'undetermined'
            if romi_left > 0:
                campaign_result = 'min ROMI > 0%'
            if romi_right < 0:
                campaign_result = 'max ROMI < 0%'
            
            
            slice_template_dict = {
                'slice_name': slice_name,
                'metric': metric,
                'romi_left': romi_left,
                'romi': romi,
                'romi_right': romi_right,
                'gross_roi_left': gross_roi_left,
                'gross_roi': gross_roi,
                'gross_roi_right': gross_roi_right,
                'st_romi_left': st_romi_left,
                'st_romi': st_romi,
                'st_romi_right': st_romi_right,
                'st_gross_roi_left': st_gross_roi_left,
                'st_gross_roi': st_gross_roi,
                'st_gross_roi_right': st_gross_roi_right,
                'st_promo_romi': st_promo_romi, 
                'st_promo_romi_left': st_promo_romi_left,
                'st_promo_romi_right': st_promo_romi_right,
                'st_promo_gross_roi': st_promo_gross_roi,
                'st_promo_gross_roi_left': st_promo_gross_roi_left,
                'st_promo_gross_roi_right': st_promo_gross_roi_right,
                'st_promo_romi_from_abs': st_promo_romi_from_abs, 
                'st_promo_romi_from_abs_left': st_promo_romi_from_abs_left,
                'st_promo_romi_from_abs_right': st_promo_romi_from_abs_right,
                'st_promo_gross_roi_from_abs': st_promo_gross_roi_from_abs,
                'st_promo_gross_roi_from_abs_left': st_promo_gross_roi_from_abs_left,
                'st_promo_gross_roi_from_abs_right': st_promo_gross_roi_from_abs_right,
                'ce': ce,
                'lt': lt,
                'rel_effect_left': rel_effect_left,
                'rel_effect': rel_effect,
                'rel_effect_right': rel_effect_right,
                'rel_effect_lt': rel_effect_lt,
                'rel_effect_lt_left': rel_effect_lt_left,
                'rel_effect_lt_right': rel_effect_lt_right,
                'abs_promo_effect': abs_promo_effect,
                'abs_promo_effect_left': abs_promo_effect_left,
                'abs_promo_effect_right': abs_promo_effect_right,
                'abs_lt_effect': abs_lt_effect,
                'abs_lt_effect_left': abs_lt_effect_left,
                'abs_lt_effect_right': abs_lt_effect_right,
                'total_elasticity': total_elasticity,
                'total_elasticity_left': total_elasticity_left,
                'total_elasticity_right': total_elasticity_right,
                'total_revenue': total_revenue,
                'budget': budget,
                'alpha': alpha,
                'campaign_result': campaign_result,
                'dtb_rel_effect': dtb_rel,
                'dtb_over_campaign': dtb_over_campaign,
                'dtb_over_year': dtb_over_year,
                'dtb_rel_effect_over_year': dtb_rel_effect_over_year,
                'dtb_rel_effect_over_year_with_lt': dtb_rel_effect_over_year_with_lt,
                'metric_over_campaign': metric_over_campaign,
                'metric_value': metric_value,
                'metric_value_left': metric_value_left,
                'metric_value_right': metric_value_right,
                'romi_from_abs_left': romi_from_abs_left,
                'romi_from_abs': romi_from_abs,
                'romi_from_abs_right': romi_from_abs_right
            }
            template_dicts.append(slice_template_dict)
#         print(template_dicts)
        return template_dicts

    
#     @classmethod
#     def get_effect_components_transfer_reg_to_fed(cls, main_params, slices_json,
#                               ts_df, metric_value_dict, ce_dict, 
#                               lt_dict, dc_results, dc_creator, metric_over_campaign_dict) -> Dict[str, Any]:
#         """
#         Used when results of regional experiment need to be extrapolated on federal campaign
#         """
#         ch_engine = dc_creator.ch_engine
#         assert ch_engine is not None
        
#         template_dicts = []
#         effects_df = cls.effectRetrieval.get_effects(ts_df)
#         slices_with_info = cls.mediaSlicesRetrieval.get_all_slices_and_info_from_config_with_dtb(
#             main_params, slices_json, ch_engine)
        
#         dtb_slices = {dtb_slice.media_slice_key: dtb_slice for dtb_slice in slices_with_info if dtb_slice.is_dtb_slice}
#         assert len(dtb_slices), 'Данных по DTB нет в dc_result, проверьте, что при загрузке в DatasetCreator указано for_media=True'

        
#         for slice_with_info in slices_with_info:
            
#             if slice_with_info.is_dtb_slice:
#                 continue # DTB is not target slice, only supplements report for target slice
                
#             slice_name, metric = slice_with_info.slice_name, slice_with_info.metric
#             budget = slice_with_info.budget
            
#             if (slice_name, metric) in metric_value_dict:
#                 total_elasticity = metric_value_dict[(slice_name, metric)]['total_elasticity']
#                 total_elasticity_std = metric_value_dict[(slice_name, metric)]['total_elasticity_std']
#             else:
#                 continue # TODO: Make normal slices with dataloader and regAB and remove this
                
#             row = effects_df.query("slice == @slice_name and metric == @metric")
#             if not len(row):
#                 continue # TODO: Make normal slices with dataloader and regAB 
                
                
#             #Get target metric data
#             metric_data = dc_results['ml_datasets'][(metric, slice_name)]['dataset']
#             metric_over_campaign = metric_over_campaign_dict[(slice_name, metric)]
            
#             # Get target metric effect
#             # Asssuming that relative effect and total elasticity is normal!
#             alpha = row["alpha"].iloc[0]
            
            
#             rel_effect = row["rel. effect"].iloc[0]
#             rel_sigma = row['rel. sigma'].iloc[0]
#             rel_effect_left = row['rel. effect lower bound'].iloc[0]
#             rel_effect_right = row['rel. effect upper bound'].iloc[0]
            
#             # Not accounting for effect correction
#             abs_promo_effect = metric_over_campaign * rel_effect # / (1 + rel_effect)
#             abs_sigma = metric_over_campaign * rel_sigma
#             abs_promo_effect_left = metric_over_campaign * rel_effect_left
#             abs_promo_effect_right = metric_over_campaign * rel_effect_right
            
            
#             # Get elasticity wrt liquidity metrics data and total revenue
#             z = norm.ppf(1 - alpha)
#             total_elasticity_left, total_elasticity_right = total_elasticity - z * total_elasticity_std, \
#                 total_elasticity + z * total_elasticity_std
#             total_revenue = metric_value_dict[(slice_name, metric)]['total_revenue']
            
#             # Get target metric value in rubles (w/o elasticity correction)
#             metric_weight = total_revenue / metric_over_campaign
#             metric_value = total_elasticity * metric_weight
#             metric_value_left = total_elasticity_left * total_revenue / metric_over_campaign
#             metric_value_right = total_elasticity_right * total_revenue / metric_over_campaign
            
#              # Get st promo estimates
#             st_promo_romi_left, st_promo_romi, st_promo_romi_right = cls.get_romi_from_elasticity_uplift(
#                 rel_effect, rel_sigma, alpha, budget, total_elasticity, total_elasticity_std, total_revenue)
#             st_promo_gross_roi_left, st_promo_gross_roi, st_promo_gross_roi_right = st_promo_romi_left + 1, \
#                 st_promo_romi + 1, st_promo_romi_right + 1
            
#             st_promo_romi_from_abs_left, st_promo_romi_from_abs, st_promo_romi_from_abs_right = cls.get_romi_from_elasticity_uplift_and_abs_effect(abs_promo_effect, abs_sigma, alpha, budget, total_elasticity, total_elasticity_std, metric_weight)
            
#             st_promo_gross_roi_from_abs = st_promo_romi_from_abs + 1
#             st_promo_gross_roi_from_abs_left = st_promo_romi_from_abs_left + 1
#             st_promo_gross_roi_from_abs_right = st_promo_romi_from_abs_right + 1
            
            
#             # Get st with ce estimates
#             if (slice_name, metric) in ce_dict:
#                 ce = ce_dict[(slice_name, metric)]
#             else:
#                 continue # TODO: Make normal slices with dataloader and regAB 

#             st_gross_roi_left, st_gross_roi, st_gross_roi_right = st_promo_gross_roi_left * ce, \
#                 st_promo_gross_roi * ce, st_promo_gross_roi_right * ce 
            
#             # Get lt estimates
#             if (slice_name, metric) in lt_dict:
#                 lt = lt_dict[(slice_name, metric)]
#             else:
#                 continue # TODO: Make normal slices with dataloader and regAB 

#             gross_roi_left, gross_roi, gross_roi_right = st_gross_roi_left * lt, st_gross_roi * lt, \
#                 st_gross_roi_right * lt
            
#             st_romi_left, st_romi, st_romi_right = st_gross_roi_left - 1, st_gross_roi - 1, \
#                 st_gross_roi_right - 1

#             romi_left, romi, romi_right = gross_roi_left - 1, gross_roi - 1, gross_roi_right - 1
            
#             abs_lt_effect = abs_promo_effect * lt
#             abs_lt_effect_left, abs_lt_effect_right = abs_promo_effect_left * lt, abs_promo_effect_right * lt
            
#             romi_from_abs_left = (st_promo_romi_from_abs_left + 1) * ce * lt - 1
#             romi_from_abs = (st_promo_romi_from_abs + 1) * ce * lt - 1 
#             romi_from_abs_right = (st_promo_romi_from_abs_right + 1) * ce * lt -1
            
#             rel_effect_lt = rel_effect * lt
#             rel_effect_lt_left, rel_effect_lt_right = rel_effect_left * lt, rel_effect_right * lt
            
            
#             # Get corresponding DTB data
#             dtb_slice = dtb_slices[slice_with_info.media_slice_key] # corresponding dtb slice
#             dtb_slice_name = dtb_slice.slice_name
#             dtb_metric_name = dtb_slice.metric
            
#             dtb_row = effects_df.query("slice == @dtb_slice_name and metric == @dtb_metric_name")
#             if len(dtb_row):
#                 dtb_rel = dtb_row['rel. effect'].iloc[0]
#                 dtb_federal_slice = dtb_slice.get_corresponding_federal_dtb_slice(ch_engine)
#                 dtb_query_slice = cls.mediaConfigToQuerySlice.get_query_slice_from_config_slice(dtb_federal_slice)
#                 dtb_fed_data = dc_creator.get_one_slice_dataset(dtb_query_slice)
#                 dtb_fed_df = dtb_fed_data[('DTB', dtb_query_slice.slice_name)]['dataset']
                
#                 dtb_over_year = cls.get_dtb_over_year(dtb_data=dtb_fed_df, test_regions=dtb_federal_slice.test_regions,
#                                       end_date=dtb_federal_slice.end_date)

#                 dtb_over_campaign = cls.get_dtb_over_campaign(dtb_data=dtb_fed_df, test_regions=dtb_federal_slice.test_regions,
#                                       start_date=dtb_federal_slice.start_date, end_date=dtb_federal_slice.end_date)

                
#                 dtb_rel_effect_over_year = dtb_rel * dtb_over_campaign / dtb_over_year
            
#                 dtb_rel_effect_over_year_with_lt = dtb_rel_effect_over_year * lt
#             else: 
#                 dtb_over_year = 0
#                 dtb_over_campaign = 0
#                 dtb_rel = 0
#                 dtb_rel_effect_over_year = 0
#                 dtb_rel_effect_over_year_with_lt = 0
                
            
#             # Get verdict
#             campaign_result = 'undetermined'
#             if romi_left > 0:
#                 campaign_result = 'min ROMI > 0%'
#             if romi_right < 0:
#                 campaign_result = 'max ROMI < 0%'
            
            
#             slice_template_dict = {
#                     'slice_name': slice_name,
#                     'metric': metric,
#                     'romi_left': romi_left,
#                     'romi': romi,
#                     'romi_right': romi_right,
#                     'gross_roi_left': gross_roi_left,
#                     'gross_roi': gross_roi,
#                     'gross_roi_right': gross_roi_right,
#                     'st_romi_left': st_romi_left,
#                     'st_romi': st_romi,
#                     'st_romi_right': st_romi_right,
#                     'st_gross_roi_left': st_gross_roi_left,
#                     'st_gross_roi': st_gross_roi,
#                     'st_gross_roi_right': st_gross_roi_right,
#                     'st_promo_romi': st_promo_romi, 
#                     'st_promo_romi_left': st_promo_romi_left,
#                     'st_promo_romi_right': st_promo_romi_right,
#                     'st_promo_gross_roi': st_promo_gross_roi,
#                     'st_promo_gross_roi_left': st_promo_gross_roi_left,
#                     'st_promo_gross_roi_right': st_promo_gross_roi_right,
#                     'st_promo_romi_from_abs': st_promo_romi_from_abs, 
#                     'st_promo_romi_from_abs_left': st_promo_romi_from_abs_left,
#                     'st_promo_romi_from_abs_right': st_promo_romi_from_abs_right,
#                     'st_promo_gross_roi_from_abs': st_promo_gross_roi_from_abs,
#                     'st_promo_gross_roi_from_abs_left': st_promo_gross_roi_from_abs_left,
#                     'st_promo_gross_roi_from_abs_right': st_promo_gross_roi_from_abs_right,
#                     'ce': ce,
#                     'lt': lt,
#                     'rel_effect_left': rel_effect_left,
#                     'rel_effect': rel_effect,
#                     'rel_effect_right': rel_effect_right,
#                     'rel_effect_lt': rel_effect_lt,
#                     'rel_effect_lt_left': rel_effect_lt_left,
#                     'rel_effect_lt_right': rel_effect_lt_right,
#                     'abs_promo_effect': abs_promo_effect,
#                     'abs_promo_effect_left': abs_promo_effect_left,
#                     'abs_promo_effect_right': abs_promo_effect_right,
#                     'abs_lt_effect': abs_lt_effect,
#                     'abs_lt_effect_left': abs_lt_effect_left,
#                     'abs_lt_effect_right': abs_lt_effect_right,
#                     'total_elasticity': total_elasticity,
#                     'total_elasticity_left': total_elasticity_left,
#                     'total_elasticity_right': total_elasticity_right,
#                     'total_revenue': total_revenue,
#                     'budget': budget,
#                     'alpha': alpha,
#                     'campaign_result': campaign_result,
#                     'dtb_rel_effect': dtb_rel,
#                     'dtb_over_campaign': dtb_over_campaign,
#                     'dtb_over_year': dtb_over_year,
#                     'dtb_rel_effect_over_year': dtb_rel_effect_over_year,
#                     'dtb_rel_effect_over_year_with_lt': dtb_rel_effect_over_year_with_lt,
#                     'metric_over_campaign': metric_over_campaign,
#                     'metric_value': metric_value,
#                     'metric_value_left': metric_value_left,
#                     'metric_value_right': metric_value_right,
#                     'romi_from_abs_left': romi_from_abs_left,
#                     'romi_from_abs': romi_from_abs,
#                     'romi_from_abs_right': romi_from_abs_right
#                 }
#             template_dicts.append(slice_template_dict)
#         return template_dicts
        
        
            
    @classmethod
    def get_romi_from_elasticity_uplift(cls, rel_eff, rel_sigma, alpha, budget, total_elasticity, 
                                        total_elasticity_std, total_revenue):
        uplift_sample = normal(loc=rel_eff, scale=rel_sigma, size=100_000)
        elasticity_sample = normal(loc=total_elasticity, scale=total_elasticity_std, size=100_000)
        
        revenue_uplift_sample = uplift_sample * elasticity_sample
        rev_rel_uplift_left = np.quantile(revenue_uplift_sample, alpha)
        rev_rel_uplift_right = np.quantile(revenue_uplift_sample, 1 - alpha)
        rev_rel_uplift = np.mean(revenue_uplift_sample)
        total_revenue_net_effect = total_revenue #/ (1 + rev_rel_uplift) # revnue w/o campaign effect
        roi_left = rev_rel_uplift_left * total_revenue_net_effect / budget
        roi = rev_rel_uplift * total_revenue_net_effect / budget
        roi_right = rev_rel_uplift_right * total_revenue_net_effect / budget
        
        return roi_left - 1, roi - 1, roi_right - 1
    
    
    @classmethod
    def get_romi_from_elasticity_uplift_and_abs_effect(cls, abs_effect, abs_sigma, alpha, budget, total_elasticity,
                                                      total_elasticity_std, metric_weight):
        
        uplift_sample = normal(loc=abs_effect, scale=abs_sigma, size=100_000)
        elasticity_sample = normal(loc=total_elasticity, scale=total_elasticity_std, size=100_000)
        
        revenue_uplift_sample = uplift_sample * elasticity_sample * metric_weight
        rev_uplift_left = np.quantile(revenue_uplift_sample, alpha)
        rev_uplift_right = np.quantile(revenue_uplift_sample, 1 - alpha)
        rev_uplift = np.mean(revenue_uplift_sample)
        roi_left = rev_uplift_left / budget
        roi = rev_uplift / budget
        roi_right = rev_uplift_right / budget
        
        return roi_left - 1, roi - 1, roi_right - 1
    

    @classmethod
    def get_dtb_over_year(cls, dtb_data: pd.DataFrame, test_regions: List[str], end_date: str) -> float:
        
        # TODO: Check all values not null during year

        calc_start_date = datetime.strptime(end_date, '%Y-%m-%d') - relativedelta(years=1)
        calc_start_date = calc_start_date.strftime("%Y-%m-%d")

        dtb_last_year_data = dtb_data[(dtb_data['date'] > calc_start_date) & (dtb_data['date'] <= end_date)]
        return dtb_last_year_data[test_regions].sum(skipna=False).sum(skipna=False)
    
    
    @classmethod
    def get_dtb_over_campaign(cls, dtb_data: pd.DataFrame, test_regions: List[str], start_date: str, end_date: str) -> float:
        
        # TODO: Check all values not null during period

        dtb_campaign_data = dtb_data[(dtb_data['date'] >= start_date) & (dtb_data['date'] <= end_date)]
        return dtb_campaign_data[test_regions].sum(skipna=False).sum(skipna=False)
    
    
    @classmethod
    def get_metric_over_campaign(cls, metric_data: pd.DataFrame, test_regions: List[str], start_date: str, end_date: str) -> float:
        
        # TODO: Check all values not null during period

        metric_data = metric_data[(metric_data['date'] >= start_date) & (metric_data['date'] <= end_date)]
        return metric_data[test_regions].sum(skipna=False).sum(skipna=False)