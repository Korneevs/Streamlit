from media.ConfigHandling.regions import regions as REGIONS


class ConfigRegionsError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class RegionsRetrieval():

    @classmethod
    def make_regions(cls, main_params):
        assert "test regions" in main_params
        assert "control regions" in main_params
        test = main_params['test regions']
        control = main_params['control regions']
        copied_regions = set(test) & set(control)
        all_regions = set(test) | set(control)
        cls._check_special_word_region(all_regions, 'Any')
        cls._check_special_word_region(test, 'Any')
        
#         self._check_special_word_region(control, 'Other')
        if len(test) >= len(REGIONS) - 2:
            raise ConfigRegionsError(f"""Вы использовали все регионы в тест. Так как байеры могут быть байерами "\
            "в разных регионах, то они засчитываются дважды. Чтобы избежать проблемы, используйте Any""")
        
        if len(copied_regions) > 0:
            raise ConfigRegionsError(f"""Повторяются регионы: {copied_regions}""")
        
        cls._check_region_spelling(all_regions)
        
        if 'Other' in control:
            control = cls._parse_other(control, test)
        if 'Other' in test:
            test = cls._parse_other(control, test)
        
        exclude_regions = set()
        if main_params.get('exclude regions'):
            exclude = set(main_params['exclude regions'])
            cls._check_region_spelling(exclude)
            cls._check_no_special_word_in_group(exclude, 'exclude', 'Other')
            cls._check_no_special_word_in_group(exclude, 'exclude', 'Any')

            control = set(control) - exclude
            test = set(test) - exclude
            exclude_regions = exclude
        if len(test) == 0:
            raise ConfigRegionsError(f"""Пустой тест!""")
  
        return {
            'test regions': test,
            'control regions': control,
            'exclude regions': exclude_regions
        }
    

    @staticmethod
    def _parse_other(array, other_array):
        final_array = set(REGIONS) - set(other_array) - set(['Any', 'Other'])
        return final_array

    
    @staticmethod
    def _check_region_spelling(array):
        bad_regs = set(array) - set(REGIONS)
        if len(bad_regs) > 0:
            raise ConfigRegionsError(f"""Нет таких регионов: {bad_regs}. Список доступных регионов: {REGIONS}""") 
        
    
    @staticmethod
    def _check_special_word_region(array, word):
        if word in array and len(array) > 1:
            raise ConfigRegionsError(f"""Нельзя смешивать {word} c другими регионами в описании test, control, exclude.""")


    @staticmethod
    def _check_no_special_word_in_group(array, ab_type, word):
        if word in array:
            raise ConfigRegionsError(f"""Нельзя ставить {word} в {ab_type}!""")

    
    @staticmethod
    def _check_region_spelling(array):
        bad_regs = set(array) - set(REGIONS)
        if len(bad_regs) > 0:
            raise ConfigRegionsError(f"""Нет таких регионов: {bad_regs}. Список доступных регионов: {REGIONS}""") 