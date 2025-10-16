import numpy as np
from datetime import datetime, timedelta


class PanelGenerator():
    
    
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed=seed)
    
    
    def sample_regions_fed_models(self, panels_list, min_group_size=20):
        validation_panels = []
        for regs, start, end in panels_list:
            regs = list(regs)
            self.rng.shuffle(regs)
            num_groups = len(regs) // min_group_size
            for i in range(num_groups):
                validation_panels.append({
                    'regions': regs[i::num_groups],
                    'start_date': datetime.strftime(start, "%Y-%m-%d"),
                    'end_date': datetime.strftime(end, "%Y-%m-%d")
                })
        return validation_panels
    
    
    def sample_regions_reg_models(self, panels_list, min_group_size=25, test_size=5, force_size=False):
        validation_panels = []
        for regs, start, end in panels_list:
            regs = list(regs)
            self.rng.shuffle(regs)
            num_groups = len(regs) // min_group_size
            for i in range(num_groups):
                group = regs[i::num_groups]
                if force_size:
                    end_of_group = min_group_size
                else:
                    end_of_group = len(group)
                validation_panels.append({
                    'test_regions': group[0:test_size],
                    'control_regions': group[test_size:end_of_group],
                    'start_date': datetime.strftime(start, "%Y-%m-%d"),
                    'end_date': datetime.strftime(end, "%Y-%m-%d")
                })
        return validation_panels
    
    
    def sample_regions_reg_models_fixed_test(self, panels_list, target_regs):
        validation_panels = []
        for regs, start, end in panels_list:
            if set(target_regs) <= set(regs):
                control_regs = list(set(regs) - set(target_regs))
                validation_panels.append({
                    'test_regions': target_regs,
                    'control_regions': control_regs,
                    'start_date': datetime.strftime(start, "%Y-%m-%d"),
                    'end_date': datetime.strftime(end, "%Y-%m-%d")
                })
        return validation_panels
    
    
    ###
    def sample_with_paired_stratification(self, panels_list, min_group_size, test_size, vertical, logical_categories, 
                                          metric, learning_period, c_engine):
        from media.media_analyser.PairedStratification import PairedStratification
        ps = PairedStratification(c_engine)
        exclude = ['Москва', "Московская область", "Санкт-Петербург", "Ленинградская область", "Краснодарский край"]
        validation_panels = []
        for regs, start, end in panels_list:
            regs = list(set(regs) - set(exclude))
            while len(regs) > min_group_size: 
                #num_groups = len(regs) // min_group_size
                test, control = ps.partition_test_control(regs, size=test_size, start=start - timedelta(learning_period),
                                                          end=start - timedelta(1), vertical=vertical[0],
                                                          logical_categories=logical_categories, metric=metric, 
                                                          exclude = exclude)
                validation_panels.append({
                    'test_regions': test.index,
                    'control_regions': control.index,
                    'start_date': datetime.strftime(start, "%Y-%m-%d"),
                    'end_date': datetime.strftime(end, "%Y-%m-%d")
                })
                
                regs = list(set(regs) - set(test.index) - set(control.index))
                
        return validation_panels
    
    
    def sample_with_fixed_test_matched_paired_control(self, panels_list, min_group_size, test_size, vertical, logical_categories, 
                                          metric, learning_period, c_engine):
        from media.media_analyser.PairedStratification import PairedStratification
        ps = PairedStratification(c_engine)
        exclude = ['Москва', "Московская область", "Санкт-Петербург", "Ленинградская область", "Краснодарский край"]
        validation_panels = []
        for regs, start, end in panels_list:
            regs = list(set(regs) - set(exclude))
            while len(regs) > min_group_size: 
                #num_groups = len(regs) // min_group_size
                self.rng.shuffle(regs)
                test_group = regs[0:test_size]
                acceptable_control = list(set(regs) - set(test_group))
                test, control = ps.match_fixed_test(test_group, acceptable_control=acceptable_control, 
                                                          start=start - timedelta(learning_period),
                                                          end=start - timedelta(1), vertical=vertical[0],
                                                          logical_categories=logical_categories, metric=metric, 
                                                          exclude = exclude)
                validation_panels.append({
                    'test_regions': test.index,
                    'control_regions': control.index,
                    'start_date': datetime.strftime(start, "%Y-%m-%d"),
                    'end_date': datetime.strftime(end, "%Y-%m-%d")
                })
                
                regs = list(set(regs) - set(test.index) - set(control.index))
                
        return validation_panels