import sys
sys.path.append('../Lib')

import os

from bxtools.vertica import VerticaEngine
import media.utils as utils


class RevenueDataAccessDma():
    """
    Класс для выгрузки из DWH значения pfp и общей выручки.
    """
    def __init__(self, engine):
        self.engine = engine

    def get_revenue_data_in_target_regions(self, vertical, logical_category, target_regions, calculation_start_date,
                                           calculation_end_date, revenue_sql_path) -> tuple:
        """
        Функция для получения pfp и общей выручки в заданном временном диапазоне, вертикалях, логкатах и регионах.
        
        :param vertical: Список интересующих вертикалей.
        :param logical_category: Список интересующих логических категорий.
        :param target_regions: Список интересующих регионов.
        :param calculation_start_date: Дата начала временного диапазона.
        :param calculation_end_date: Дата конца временного диапазона.
        :param revenue_sql_path: Название sql-файла с запросом.
        
        :return: Кортеж из двух чисел - pfp выручки и общей выручки.
        """
        # Составляем условия фильтрации расчета выручки
        vertical_condition = self._get_vertical_condition_text(vertical)
        logical_category_condition = self._get_logical_category_condition_text(logical_category)
        target_regions_condition = self._get_target_regions_condition_text(target_regions)

        # Заполняем параметры sql-запроса полученными значениями и исполняем его
        params = {
            'calculation_start_date': f"'{calculation_start_date}'",
            'calculation_end_date': f"'{calculation_end_date}'",
            'vertical_condition': vertical_condition,
            'logical_category_condition': logical_category_condition,
            'target_regions_condition': target_regions_condition
        }
        path = revenue_sql_path
        revenue_query_template = utils.read_sql_file(os.path.abspath(path))
        revenue_query_filled = utils.fill_query(revenue_query_template, params)
        revenue = self.engine.select(revenue_query_filled)
        
        return revenue.iloc[0, 0], revenue.iloc[0, 1]    
    
    def _get_vertical_condition_text(self, vertical) -> str:
        """
        Функция для формирования условия на вертикали при расчете выручки.
        
        :param vertical: Список интересующих вертикалей.
        
        :return: Строка с фильтром для sql-запроса.
        """
        # Если интересуют любые вертикали, то возвращаем фиктивное условие
        if 'Any' in vertical:
            return 'True'
        else: 
            vertical_list = ', '.join([f"'{vert}'" for vert in vertical])
            # Немного разная логика работы sql-запроса на Trino и Vertica
            if isinstance(self.engine, VerticaEngine):
                return f"lc.vertical in ({vertical_list})" 
            else:
                return f"vertical in ({vertical_list})" 

    def _get_logical_category_condition_text(self, logical_category) -> str:
        """
        Функция для формирования условия на логические категориии при расчете выручки.
        
        :param logical_category: Список интересующих логкатов.
        
        :return: Строка с фильтром для sql-запроса.
        """
        # Если интересуют любые логкаты, то возвращаем фиктивное условие
        if 'Any' in logical_category:
            return 'True'
        else: 
            logical_category_list = ', '.join([f"'{lc}'" for lc in logical_category])
            # Немного разная логика работы sql-запроса на Trino и Vertica
            if isinstance(self.engine, VerticaEngine):
                return f"lc.logical_category in ({logical_category_list})" 
            else:
                return f"logical_category in ({logical_category_list})" 

    def _get_target_regions_condition_text(self, target_regions) -> str:
        """
        Функция для формирования условия на регионы при расчете выручки.
        
        :param target_regions: Список интересующих регионов.
        
        :return: Строка с фильтром для sql-запроса.
        """
        # Если интересуют любые регионы, то возвращаем фиктивное условие
        if 'Any' in target_regions:
            return 'True'
        else: 
            target_regions_list = ', '.join([f"'{reg}'" for reg in target_regions])
            # Немного разная логика работы sql-запроса на Trino и Vertica
            if isinstance(self.engine, VerticaEngine):
                return f"l.Region in ({target_regions_list})"
            else:
                return f"region in ({target_regions_list})"
