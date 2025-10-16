import sys
sys.path.append('../Lib')

import os

from bxtools.vertica import VerticaEngine
from media.MetricValue.elasticity_params_config import VERTICAL_NPFP_ELASTICITIES, VERTICAL_NPFP_ELASTICITY_STDS
from media.MetricValue.RevenueDataAccessDMA import RevenueDataAccessDma
from media.ConfigHandling.SlicesRetrieval import MediaSlicesRetrieval



class MetricValueCalculation():
    """
    Класс для вычисления ценности short-term эффекта в метрике.
    """
    def __init__(self, engine):        
        self.engine = engine

        # Вспомогательные классы для вычисления
        self.revenueDataAccessDma = RevenueDataAccessDma(engine)
        self.mediaSlicesRetrieval = MediaSlicesRetrieval
        
        
    def get_liquidity_metric_values(self, main_params, slices_json, revenue_sql_path=None) -> dict:
        """
        Функция для расчета ценности метрик (эластичности по выручке)
        
        :param main_params: Словарь с основными параметрами РК.
        :param slices_json: Словарь с анализируемыми срезами.
        :param revenue_sql_path: Путь к sql-файлу с расчетом pfp и общей выручки.
        
        :return: Словарь с коэффициентами эластичности, std эластичности и общей выручкой по каждому срезу
        """
        # Если путь до sql-файла не был передан, то подтягиваем его по умолчанию
        if revenue_sql_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            # Немного разная логика расчетов для Trino и Vertica
            if isinstance(self.engine, VerticaEngine):
                revenue_sql_path = os.path.join(module_dir, 'pfp_and_all_revenue.sql')
            else:
                revenue_sql_path = os.path.join(module_dir, 'pfp_and_all_revenue_trino.sql')
        
        # Создаем словарь для избежания повторных расчетов ценности. 
        metric_value_info = {} 
        # Словарь с информацией по всем срезам
        slice_metric_value_info = {}

        # Достаем параметры срезов
        slices_with_info = self.mediaSlicesRetrieval.get_all_slices_and_info_from_config(
            main_params, 
            slices_json)

        # Последовательно обрабатываем каждый срез
        for slice_with_info in slices_with_info:
            # Скипаем DTB-срезы (расчет настроен по buyers)
            if slice_with_info.is_dtb_slice:
                continue

            # Парсим всю необходимую инфу из среза
            slice_name, metric = slice_with_info.slice_name, slice_with_info.metric 
            vertical, logical_category = slice_with_info.vertical, slice_with_info.logical_category
            target_regions = slice_with_info.test_regions
            calculation_start_date, calculation_end_date = slice_with_info.start_date, slice_with_info.end_date
            
            # Составляем ключ для словаря metric_value_info
            key = self.get_slice_key(vertical, logical_category, target_regions, calculation_start_date, calculation_end_date)
            # Если еще не считали ценность для среза с такими параметрами, то запускаем расчет
            if key not in metric_value_info:
                value_info = {}
                # Расчитываем необходимые значения и добавляем их в словарь
                total_elasticity, total_el_std, total_revenue = self.get_total_elasticity_and_revenue(
                    vertical, 
                    logical_category, 
                    target_regions, 
                    calculation_start_date,
                    calculation_end_date, 
                    revenue_sql_path
                )
                value_info['total_elasticity'] = total_elasticity
                value_info['total_elasticity_std'] = total_el_std
                value_info['total_revenue'] = total_revenue
                # Запоминаем посчитанные значения по параметрам среза
                metric_value_info[key] = value_info
            
            # Запоминаем значения для среза
            slice_metric_value_info[(slice_name, metric)] = metric_value_info[key]
            
        return slice_metric_value_info
    
    
    def get_total_elasticity_and_revenue(self, vertical, logical_category, target_regions, calculation_start_date,
                                         calculation_end_date, revenue_sql_path) -> tuple:
        """
        Функция для получения значения эластичности, ее std и общей выручки для конкретного среза.
        
        :param vertical: Список с вертикалями среза.
        :param logical_category: Список с логкатами среза.
        :param target_regions: Список с регионами среза.
        :param calculation_start_date: Дата начала анализируемого периода.
        :param calculation_end_date: Дата конца анализируемого периода.
        :param revenue_sql_path: Путь к sql-файлу с расчетом pfp и общей выручки (с учетом фильтраций на даты, вертикали, логкаты
        и регионы).
        
        :return: Кортеж из трех значений - общая эластичность, std общей эластичности, общая выручка.
        """
        # Получаем значения pfp и общей выручки
        pfp_revenue, total_revenue = self.revenueDataAccessDma.get_revenue_data_in_target_regions(
            vertical, 
            logical_category, 
            target_regions,
            calculation_start_date,
            calculation_end_date,
            revenue_sql_path
        )
        
        # Находим значение для эластичности и его std
        total_elasticity, total_el_std = self.get_total_elasticity_from_revenue(pfp_revenue, total_revenue, vertical)
        
        return total_elasticity, total_el_std, total_revenue
    
    
    def get_total_elasticity_from_revenue(self, pfp_revenue, total_revenue, vertical=['Any']) -> tuple:
        """
        Функция для получения общей эластичности (и стандартного отклонения для нее) по значениям выручки.
        
        :param pfp_revenue: Значение pfp выручки.
        :param total_revenue: Значение общей выручки.
        :param vertical: Название вертикали (в списке) для расчета эластичности.
        
        :return: Кортеж из значения общей эластичности и его станд. отклонения
        """
        # Вычисляем долю pfp выручки в общей
        pfp_share = pfp_revenue / total_revenue
        # Достаем значения эластичности для npfp выручки по вертикали
        npfp_elasticity, npfp_elast_std = self._get_npfp_elasticity(vertical)
        # Расчитываем общую эластичность (с учетом долей pfp и npfp)
        total_elasticity = pfp_share * 1 + (1 - pfp_share) * npfp_elasticity
        # Корректируем std у npfp выручки на ее долю
        total_el_std = (1 - pfp_share) * npfp_elast_std
        
        return total_elasticity, total_el_std
            
    @staticmethod
    def _get_npfp_elasticity(vertical) -> tuple:
        """
        Статичный метод для получения значений эластичности (выручки по метрике) для конкретной вертикали.
        
        :param vertical: Список с одной (!) вертикалью.
        
        :return: Кортеж из двух значений эластичности: по pfp и общей выручке
        """
        # Проверки, что передали корректную вертикаль в правильном формате
        assert len(vertical) == 1
        assert vertical[0] in VERTICAL_NPFP_ELASTICITIES, vertical[0]
        assert vertical[0] in VERTICAL_NPFP_ELASTICITY_STDS, vertical[0]
        
        return VERTICAL_NPFP_ELASTICITIES[vertical[0]], VERTICAL_NPFP_ELASTICITY_STDS[vertical[0]]
    
    @staticmethod
    def get_slice_key(vertical, logical_category, target_regions, calculation_start_date, calculation_end_date) -> tuple:
        """
        Статичный метод для получения ключа для словаря metric_value_info.
        
        :param vertical: Список вертикалей.
        :param logical_category: Список логкатов.
        :param target_regions: Список регионов.
        :param calculation_start_date: Дата начала периода.
        :param calculation_end_date: Дата конца периода.
        
        :return: Кортеж-ключ для словаря.
        """
        return (tuple(sorted(list(set(vertical)))), 
                tuple(sorted(list(set(logical_category)))), 
                tuple(sorted(list(set(target_regions)))),
                calculation_start_date,
                calculation_end_date)
    