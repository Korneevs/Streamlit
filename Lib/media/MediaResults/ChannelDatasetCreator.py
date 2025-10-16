import sys
import os
import inspect
sys.path.append('../Lib')

import pandas as pd
import numpy as np
import copy
import datetime as dt
from decimal import Decimal, ROUND_HALF_UP

from media.DatasetWorker.DatasetCreator import DatasetCreator
from media.MetricValue.MetricValueCalculation import MetricValueCalculation
from media.CrossEffects.CrossEffectsForSlices import CrossEffectsForSlices
from media.LongTermEffects.LTEffect import LongTermEffect
from media.Visualizer.BeautyUnderneath import TableResults
from media.ConfigHandling.SlicesRetrieval import MediaSlicesRetrieval
from media.MediaResults.get_cross_effects import vertical_cross_effects_dict

import media.utils as utils

class ChannelDatasetCreator:
    def __init__(self, c_engine, t_engine):
        """
        Инициализация класса
        :param c_engine: Подключение к ClickHouse.
        :param t_engine: Подключение к Trino
        """
        self.c_engine = c_engine
        self.t_engine = t_engine
        
    def get_sql_query(self, sql_file_name):
        """
        Получаем sql запрос - сами запросы лежат в папке SQL
        """
        # Загружаем SQL‑файл с запросами.
        sql_file_path = os.path.abspath(os.path.join(
            os.path.dirname(inspect.getfile(self.__class__)),
            'SQL',
            sql_file_name
        ))

        # Читаем сам SQL
        query_template = utils.read_sql_file(sql_file_path)
        
        return query_template
    
    def expand_regions(self, region_str):
        """
        Функция для замены названий регионов
        :region_str: Данные флайта, для которого производится замена
        :return: Измененная строка регионов
        """
        # Словарь для замены названий регионов
        region_map = {'МО': 'Московская область', 'ЛО': 'Ленинградская область'}

        regions = [r.strip() for r in region_str.split(',')]
        replaced_regions = [region_map.get(region, region) for region in regions]

        return ", ".join(replaced_regions)
    
    def convert_metric(self, metric):
        """
        Функция для форматирования метрики
        Пример: 0.5% -> 0.005
        """
        return float(metric.strip().replace("%", ""))/100
    

    def get_dataset_from_ps(self, channel: str, flight_name) -> pd.DataFrame:
        """
        Берем все необходимые данные от People & Screens
        :flight_name: Название анализируемого флайта
        :channel: Канал (TV, Digital, OOH или Other)
        :return: Возвращаем таблицу с информацией о флайте
        """
        channel = channel.lower()
        # Замена названий каналов
        channel_name = {
            'tv': ['TV Sponsorship', 'Reg TV', 'Nat TV'],
            'digital': ['Internet'],
            'ooh': ['OOH'],
            'other': ['Radio','Indoor','Press']
        }
        if channel not in channel_name:
            raise ValueError(f"Неизвестный тип канала: {channel}. Допустимые значения: ТV, Digital, OOH, Other")
            
        sql_channel_list = ", ".join(f"'{ch}'" for ch in channel_name[channel])

        query_template = self.get_sql_query('flight_info.sql')
        
        # Фильтр для поиска по конкретному флайту или всем флайтам
        extra_filter = ""
        if flight_name: 
            safe_name = flight_name.replace("'", "''") # экранизируем апострофы
            extra_filter = f"AND adv_campaign = '{safe_name}'"
        else:                         
            extra_filter = ""
        
        # Подставляем параметры в SQL-шаблон
        query = query_template.format(
            channel=channel,
            sql_channel_list=sql_channel_list,
            extra_filter=extra_filter
        )
        df = self.t_engine.select(query)
        
        # Приводим регионы к допустимому формату
        df['region'] = df['region'].apply(self.expand_regions)

        return df
    
    def get_flight_budget(self, flight_name):
        """
        Получаем информацию о бюджете флайта по всем каналам
        """
        query_template = self.get_sql_query('get_flight_budget.sql')
        
        extra_filter = ""
        if flight_name:                    
            safe_name = flight_name.replace("'", "''") # экранизируем апострофы
            extra_filter = f"AND adv_campaign = '{safe_name}'"
        else:                         
            extra_filter = ""
        
        # Подставляем параметры в SQL-шаблон
        query = query_template.format(
            extra_filter=extra_filter
        )
        df = self.t_engine.select(query)
        
        return df
    
    def get_slice_metrics(self, ts_df, tb_df, metric_name, slice_name):
        """
        Функция для получения информации о relative uplift и relative mde для выбранной метрики
        в выбранной категории
        """
        # Строка с информацией о метриках из TableResults
        goal_row = tb_df[(tb_df["metric"]==metric_name) & (tb_df["slice"]==slice_name)]
        
        metric_uplift = goal_row["rel. effect"].iloc[0].split("<br>")[0]
        
        rel_mde_col = goal_row.columns[
            goal_row.columns.str.contains(r'rel\.?\s*MDE',  
                                           case=False,
                                           regex=True)
        ][0]

        metric_mde = goal_row[rel_mde_col].iloc[0]
        
        return {"slice_name": slice_name,
                "metric_rel_uplift": self.convert_metric(metric_uplift),
                "metric_rel_mde": self.convert_metric(metric_mde)}

    def get_metric_abs_uplift(self, df):
        return df["metric_rel_uplift"] * df["sum_metric_per_flight"]
    
    def get_metric_abs_mde(self, df):
        return df["metric_rel_mde"] * df["sum_metric_per_flight"]
    
    def get_main_metric_abs_uplift(self, df):
        return df["metric_rel_uplift"] * df["sum_main_metric_per_flight"]
    
    def get_main_metric_abs_mde(self, df):
        return df["metric_rel_mde"] * df["sum_main_metric_per_flight"]
    
    def get_revenue(self, df):
        return df["metric_abs_uplift"] * df["metric_to_revenue"] * df["long_term_effect"] * df["cross_effect"]

    def get_metric_to_revenue(self, df):
        return df["elasticity"] * df["sum_revenue_per_flight"] / df["sum_metric_per_flight"]

    def get_metric_cost_per(self, df):
        metric_abs_uplift = df["metric_main_abs_uplift"].astype(float)
        return np.where(metric_abs_uplift == 0, -1, df["channel_budget_analysed"].astype(float) / metric_abs_uplift)

    def get_romi(self, df):
        return df['revenue'] / df["channel_budget_analysed"] - 1

    def get_mde_romi(self, df):
        return ((df["metric_abs_mde"] * df["metric_to_revenue"]) / df["channel_budget_analysed"]) * \
               df["long_term_effect"] * df["cross_effect"]

    def get_verdict(self, romi, mde_romi):
        if (mde_romi + romi) < 0:
            return 'ROMI < 0%'
        return 'campaign gray'

    def get_params(self, main_params, slices, main_slice_name) -> dict:
        """
        Функция для вычисления основных метрик
        """
        main_params = copy.deepcopy(main_params)

        # ROMI считаем по proxy metric
        main_params['metrics'] = [main_params['proxy_metric'], main_params['main_metric']]
        
        dc = DatasetCreator(self.c_engine, self.t_engine, for_media=True)
        dc_result = dc.get_datasets(main_params, slices)
        
        # Кросс эффект
        ce_dict = {}
        raw_vertical = slices[main_slice_name]['vertical']
        vertical = raw_vertical[0] if isinstance(raw_vertical, list) else raw_vertical

        # берем фиксированные крос эффекты
        if vertical in vertical_cross_effects_dict:
            coeff = vertical_cross_effects_dict[vertical]
            for metric in main_params['metrics']:
                ce_dict[(main_slice_name, metric)] = coeff
        else:
            cefs = CrossEffectsForSlices(self.t_engine)
            ce_dict = cefs.get_crosseffects_for_slices(main_params, slices)
        
        mvc = MetricValueCalculation(self.t_engine)
        elasticity_data = mvc.get_liquidity_metric_values(main_params, slices)
        
        # Long-term эффект
        lte = LongTermEffect()
        lt_dict = lte.get_lt_coef(main_params, slices)
        
        
        # Сумма прокси метрики в указанный период 
        curr_df = dc_result['ml_datasets'][(main_params['proxy_metric'], main_slice_name)]['dataset']
        sum_metric_per_flight = curr_df[
            (curr_df['date'] >= main_params['analysed_start_date']) &
            (curr_df['date'] <= main_params['analysed_end_date'])
        ][main_params['test regions']].sum().sum()
        
        # Сумма целевой метрики в указанный период
        curr_df_main = dc_result['ml_datasets'][(main_params['main_metric'], main_slice_name)]['dataset']
        sum_main_metric_per_flight = curr_df_main[
            (curr_df_main['date'] >= main_params['analysed_start_date']) &
            (curr_df_main['date'] <= main_params['analysed_end_date'])
        ][main_params['test regions']].sum().sum()
        
        key = next(iter(elasticity_data))

        return {
            "elasticity": elasticity_data[key]['total_elasticity'],
            "sum_revenue_per_flight": elasticity_data[key]['total_revenue'],
            "long_term_effect": lt_dict[key],
            "cross_effect": ce_dict[key],
            "sum_metric_per_flight": sum_metric_per_flight,
            "sum_main_metric_per_flight": sum_main_metric_per_flight
        }
    
    def make_final_row_results(self, flight_info, custom_metrics):
        """
        Функция для вывода таблицы с полученными результатами
        :flight_info: Таблица с основной информацией о флайтах
        :return: Итоговая таблица
        """
        flight_info["main_log_cats"] = flight_info.apply(
             lambda row: row["slices"][row["main_slice_name"]]["logical_category"],
             axis=1
         )
        
        # Приводим даты в необходимый формат
        date_cols = ["date_start", "date_end", "analysed_date_start", "analysed_date_end"]
        for dc in date_cols:
            flight_info[dc] = pd.to_datetime(flight_info[dc]).dt.date
        
        # Приводим числовые значения в необходимый формат
        decimal_cols = [
            "flight_budget", "channel_budget", "channel_budget_analysed", "metric_rel_uplift", "metric_rel_mde","metric_abs_uplift", "metric_abs_mde",
            "revenue", "metric_to_revenue", "elasticity", "cross_effect",
            "long_term_effect", "sum_metric_per_flight", "cost_per_metric", "romi", "mde_romi"
        ]
        
        for dc in decimal_cols:
            flight_info[dc] = flight_info[dc].apply(
                lambda x: np.NaN if pd.isna(x) else Decimal(str(x))
            )
        
        int_cols = {
            "flight_budget",
            "channel_budget",
            "channel_budget_analysed",
            "metric_abs_uplift",
            "metric_abs_mde",
            "revenue",
            "metric_to_revenue",
            "sum_metric_per_flight",
            "cost_per_metric",
        }
        dec4_cols = set(decimal_cols) - int_cols
        # до целого
        flight_info[list(int_cols)] = flight_info[list(int_cols)].applymap(
            lambda x: x if pd.isna(x) else x.quantize(Decimal("1"), ROUND_HALF_UP)
        )
        # до 4-х знаков после запятой
        flight_info[list(dec4_cols)] = flight_info[list(dec4_cols)].applymap(
            lambda x: x if pd.isna(x) else (
                x.quantize(Decimal("0.0001"), ROUND_HALF_UP).normalize()
            )
        )
        
        final_df = pd.DataFrame({
            "flight_name": flight_info["flight_name"],
            "vertical": flight_info["vertical"],
            "category": flight_info["main_slice_name"],
            "type": flight_info["type"],
            "tier": flight_info["tier"],
            "channel": flight_info["channel"],
            "date_start": flight_info["date_start"],
            "date_end": flight_info["date_end"],
            "analysed_date_start": flight_info["analysed_date_start"],
            "analysed_date_end": flight_info["analysed_date_end"],
            "flight_budget": flight_info["flight_budget"],
            "channel_budget": flight_info["channel_budget"],
            "channel_budget_analysed": flight_info["channel_budget_analysed"],
            "analysis_tool": flight_info['analysis_tool'],
            "main_metric": flight_info["main_metric"],
            "is_main_metric": flight_info["is_main_metric"],
            "proxy_metric": flight_info["proxy_metric"],
            "rel_metric_uplift": flight_info["metric_rel_uplift"],
            "rel_metric_mde": flight_info["metric_rel_mde"],
            "abs_metric_uplift": flight_info["metric_main_abs_uplift"],
            "abs_metric_mde": flight_info["metric_main_abs_mde"],
            "romi": flight_info["romi"],
            "mde_romi": flight_info["mde_romi"],
            "revenue": flight_info["revenue"],
            "metric_to_revenue": flight_info["metric_to_revenue"],
            "metric_to_revenue_elasticity": flight_info["elasticity"],
            "cross_effect": flight_info["cross_effect"],
            "long_term_effect": flight_info["long_term_effect"],
            "sum_metric_per_flight": flight_info["sum_main_metric_per_flight"],
            "cost_per_metric": flight_info["cost_per_metric"],
            "verdict": pd.Series(np.vectorize(self.get_verdict)(flight_info["romi"], flight_info["mde_romi"]), index=flight_info.index),
            "main_log_cats": flight_info["main_log_cats"],
            "cf_link": flight_info["cf_link"],
            "custom_slice_name_1": custom_metrics.get(f"custom_slice_name_1", {}).get("slice_name"),
            "custom_slice_uplift_1": custom_metrics.get(f"custom_slice_name_1", {}).get("metric_rel_uplift"),
            "custom_slice_mde_1": custom_metrics.get(f"custom_slice_name_1", {}).get("metric_rel_mde"),
            "custom_slice_name_2": custom_metrics.get(f"custom_slice_name_2", {}).get("slice_name"),
            "custom_slice_uplift_2": custom_metrics.get(f"custom_slice_name_2", {}).get("metric_rel_uplift"),
            "custom_slice_mde_2": custom_metrics.get(f"custom_slice_name_2", {}).get("metric_rel_mde"),
            "custom_slice_name_3": custom_metrics.get(f"custom_slice_name_3", {}).get("slice_name"),
            "custom_slice_uplift_3": custom_metrics.get(f"custom_slice_name_3", {}).get("metric_rel_uplift"),
            "custom_slice_mde_3": custom_metrics.get(f"custom_slice_name_3", {}).get("metric_rel_mde"),
            "custom_slice_name_4": custom_metrics.get(f"custom_slice_name_4", {}).get("slice_name"),
            "custom_slice_uplift_4": custom_metrics.get(f"custom_slice_name_4", {}).get("metric_rel_uplift"),
            "custom_slice_mde_4": custom_metrics.get(f"custom_slice_name_4", {}).get("metric_rel_mde")
        })
        
        return final_df
        

#     def get_tv_channel_table(self, media_df, is_main_metric=True):
#         """
#         Функция временная - нужна для того, чтобы перенести уже имеющуюся в вертике информацию о тв флайтах
#         """
#         ps_flight_info = self.get_dataset_from_ps('TV', flight_name="")
        
#         ps_flight_info['date_start'] = pd.to_datetime(ps_flight_info['date_start'])
#         ps_flight_info['date_end'] = pd.to_datetime(ps_flight_info['date_end'])
        
#         ps_flight_info = ps_flight_info[
#         (ps_flight_info["date_start"] > "2025-01-01") &
#         (ps_flight_info["date_end"]   < "2025-06-01")
#     ][["flight_name", "date_start", "date_end", "budget"]]
        
#         flight_budget = self.get_flight_budget(flight_name="")
#         df = ps_flight_info.merge(flight_budget, how='inner', on='flight_name')
        
#         media_df["metric_mde"]        = pd.to_numeric(media_df["metric_mde"], errors="coerce")
#         media_df["metric_cat_flight"] = pd.to_numeric(media_df["metric_cat_flight"], errors="coerce")
        
#         media_df.rename(columns={
#             "flight": "flight_name",
#             "date_start": "analysed_date_start",
#             "date_end": "analysed_date_end"
#             }, inplace=True)
        
#         media_df = media_df.merge(df, how='left', on='flight_name')
#         final_df = pd.DataFrame({
#             "flight_name": media_df["flight_name"],
#             "vertical": media_df["vertical"],
#             "category": media_df["category"],
#             "type": media_df["type"],
#             "channel": 'TV',
#             "date_start": media_df["date_start"].combine_first(media_df["analysed_date_start"]),
#             "date_end":  media_df["date_end"].combine_first(media_df["analysed_date_end"]),
#             "analysed_date_start": media_df["analysed_date_start"],
#             "analysed_date_end": media_df["analysed_date_end"],
#             "tier": media_df["tier"],
#             "flight_budget":  media_df["flight_budget"].combine_first(media_df["budget_mln"]*1000000),
#             "channel_budget": media_df["budget"].combine_first(media_df["analyzed_budget_mln"]*1000000),
#             "channel_budget_analysed": media_df["analyzed_budget_mln"]*1000000,
#             "analysis_tool": media_df['analysis_tool'],
#             "main_metric": media_df["metric"],
#             "is_main_metric": is_main_metric,
#             "proxy_metric": media_df["goal"],
#             "rel_metric_uplift": media_df["metric_uplift"]/100,
#             "rel_metric_mde": media_df["metric_mde"]/100,
#             "abs_metric_uplift": media_df["metric_abs"],
#             "abs_metric_mde": media_df["metric_mde"] * media_df["metric_cat_flight"],
#             "romi": media_df["romi"]/100,
#             "mde_romi": media_df["mde_romi"]/100,
#             "revenue": (media_df["romi"]/100+1)*media_df["analyzed_budget_mln"]*1000000,
#             "metric_to_revenue": media_df["metric_value"],
#             "metric_to_revenue_elasticity": media_df["elasticy"]/100,
#             "cross_effect": media_df["cross_effect"],
#             "long_term_effect": 1.95,
#             "sum_metric_per_flight": media_df["metric_cat_flight"],
#             "cost_per_metric": media_df['metric_cost_per'],
#             "verdict": media_df['verdict'],
#             "main_log_cats": media_df["main_log_cats"],
#             "cf_link": media_df["cf_link"],
#             "custom_slice_name_1": None,
#             "custom_slice_uplift_1": None,
#             "custom_slice_mde_1": None,
#             "custom_slice_name_2": None,
#             "custom_slice_uplift_2": None,
#             "custom_slice_mde_2": None,
#             "custom_slice_name_3": None,
#             "custom_slice_uplift_3": None,
#             "custom_slice_mde_3": None,
#             "custom_slice_name_4": None,
#             "custom_slice_uplift_4": None,
#             "custom_slice_mde_4": None
#         })
#         
#         
#         return final_df   
    
    
    def make_flight_row_for_channel(
        self,
        main_params: dict,
        slices: dict,
        slice_to_dash_name: dict,
        ts_df: pd.DataFrame,
        channel: str,
        is_main_metric=True
    ) -> pd.DataFrame:

        rows = []
        target_metric = main_params["main_metric"]
        
        metrics_list = [m for m in main_params["metrics"] if m != main_params['main_metric']]
        if target_metric not in metrics_list:
            metrics_list.append(target_metric)
        
        for metric in metrics_list:
            mp = copy.deepcopy(main_params)
            is_target = (metric == target_metric) & is_main_metric
            if not is_target:
                mp["main_metric"] = metric
                mp["proxy_metric"] = metric

            row_df = self.make_single_metric_row_for_channel(
                mp,
                channel,
                slices,
                slice_to_dash_name,
                ts_df,
                is_main_metric=is_target,
            )
            rows.append(row_df)
        
        return pd.concat(rows, ignore_index=True)
    
    def make_single_metric_row_for_channel(self, main_params, channel, slices, slice_to_dash_name, ts_df, is_main_metric=True):
        """
        Функция для получения информации о флайте
        
        :main_params: Основные параметры флайта
        :slices: Рассматриваемые срезы
        :slice_to_dash_name: Словарь с информацией о целевом и кастомными срезами
        
        :return: Словарь с информацией о флайте
        """
        # Получаем информацию от People&Screens
        flight_name = main_params["flight_name"]
        ps_flight_info = self.get_dataset_from_ps(channel, flight_name)
        ps_flight_info["tier"] = (
                ps_flight_info["tier"]               
                    .replace({"Tier 1": "T1",
                              "Tier 2": "T2",
                              "Tier 3": "T3"})
            )
        # Получаем инфомацию об общем бюджете флайтов
        flight_budget = self.get_flight_budget(flight_name)
        
        tb = TableResults()
        tb_df = tb.get_raw_dataframe(ts_df, negative_metrics=[], elasticity_dict=None)
        
        # Получаем информацию о relative uplift и relative mde для прокси метрики в выбранных срезах
        custom_metrics = {
            custom_name: self.get_slice_metrics(ts_df, tb_df, main_params['proxy_metric'], slice_name)
            for custom_name, slice_name in slice_to_dash_name.items()
        }
        
        main_slice_name = custom_metrics["main"]['slice_name']
                
        flight_info = pd.DataFrame({
            "flight_name": main_params["flight_name"],
            "vertical": ps_flight_info["vertical"],
            "channel": main_params["channel"],
            "tier": ps_flight_info["tier"],
            "type": ps_flight_info["adv_type"],
            "date_start": main_params["flight_start_date"],
            "date_end": main_params["flight_end_date"],
            "analysed_date_start": main_params["analysed_start_date"],
            "analysed_date_end": main_params["analysed_end_date"],
            "flight_budget": flight_budget["flight_budget"],
            "channel_budget": ps_flight_info["budget"],
            "channel_budget_analysed": main_params["flight_budget"],
            "main_metric": main_params["main_metric"],
            "is_main_metric": is_main_metric,
            "proxy_metric": main_params["proxy_metric"],
            "analysis_tool": ts_df["method"].unique()[0],
            "metric_rel_uplift": custom_metrics["main"]['metric_rel_uplift'],
            "metric_rel_mde": custom_metrics["main"]['metric_rel_mde'],
            "main_params": [main_params],
            "main_slice_name": main_slice_name,
            "slices": [{main_slice_name: slices[main_slice_name]}],
            "cf_link": None
        })
        
        # Считаем эластичность, сумму выручки за флайт, лонг терм эффект, кросс эффект, сумму целевой метрики 
        if is_main_metric:
            params_dict = flight_info.apply(
                lambda row: pd.Series(
                    self.get_params(row["main_params"], row["slices"], row["main_slice_name"])
                ),
                axis=1,
            )
            flight_info = flight_info.join(params_dict)

            flight_info["metric_abs_uplift"] = self.get_metric_abs_uplift(flight_info)
            flight_info["metric_abs_mde"] = self.get_metric_abs_mde(flight_info)
            flight_info["metric_main_abs_uplift"] = self.get_main_metric_abs_uplift(flight_info)
            flight_info["metric_main_abs_mde"] = self.get_main_metric_abs_mde(flight_info)
            flight_info["metric_to_revenue"] = self.get_metric_to_revenue(flight_info)
            flight_info["revenue"] = self.get_revenue(flight_info)
            flight_info["cost_per_metric"] = self.get_metric_cost_per(flight_info)

            flight_info["main_log_cats"] = flight_info.apply(
                lambda row: row["slices"][row["main_slice_name"]]["logical_category"], axis=1
            )

            flight_info["romi"] = self.get_romi(flight_info)
            flight_info["mde_romi"] = self.get_mde_romi(flight_info)

            return self.make_final_row_results(flight_info, custom_metrics)
        
        for metric in ["metric_abs_uplift", "metric_abs_mde","metric_main_abs_uplift", "metric_main_abs_mde", "metric_to_revenue", "revenue",
                       "cost_per_metric", "romi", "mde_romi", "elasticity", "cross_effect",
                       "long_term_effect", "sum_metric_per_flight", "sum_main_metric_per_flight", "metric_to_revenue_elasticity",
    "verdict"]:
            flight_info[metric] = np.nan

        return self.make_final_row_results(flight_info, custom_metrics)
   

    def update_channel_table(self, df, channel):
        """
        Функция для добавления информации в DWH
        """
        df = copy.deepcopy(df)
        # Название public таблицы по каналу
        table_name = f"public.media_{channel.lower()}_results_trino"
        
        # Приводим даты к нужному формату
        for c in ["date_start", "date_end",
                  "analysed_date_start", "analysed_date_end"]:
            df[c] = pd.to_datetime(df[c]).dt.date

        # Вычисляем последний id
        next_id = (
            self.t_engine.select(
                f"SELECT COALESCE(MAX(id),0) FROM {table_name}"
            ).iloc[0, 0] + 1
        )
        
        # Добавляем информацию о времени добавления и логина пользователя
        now_ts = dt.datetime.utcnow()               
        user = self.t_engine.conn_info.get("user")

        df["id"]= pd.Series(
            range(next_id, next_id + len(df)), dtype="int64"
        )
        df["version_stamp"] = now_ts         
        df["created_by"] = user
        
        df["main_log_cats"] = df["main_log_cats"].apply(
            lambda x: ", ".join(x) if isinstance(x, (list, tuple)) else x
        )
        
        trino_cols = [
            "id", "flight_name", "vertical", "category", "type", "channel",
            "date_start", "date_end", "analysed_date_start", "analysed_date_end",
            "tier", "flight_budget", "channel_budget", "channel_budget_analysed", "analysis_tool",
            "main_metric", "is_main_metric", "proxy_metric", "rel_metric_uplift", "rel_metric_mde",
            "abs_metric_uplift", "abs_metric_mde", "romi", "mde_romi", "revenue",
            "metric_to_revenue", "metric_to_revenue_elasticity", "cross_effect",
            "long_term_effect", "sum_metric_per_flight", "cost_per_metric",
            "verdict", "main_log_cats", "cf_link", "custom_slice_name_1", "custom_slice_uplift_1",
            "custom_slice_mde_1", "custom_slice_name_2", "custom_slice_uplift_2",
            "custom_slice_mde_2", "custom_slice_name_3", "custom_slice_uplift_3",
            "custom_slice_mde_3", "custom_slice_name_4", "custom_slice_uplift_4",
            "custom_slice_mde_4", "created_by", "version_stamp"
        ]
        df = df[trino_cols]
        
        self.t_engine.insert(df, table_name)
        
    def create_channel_table(self, channel):
        """
        Создаем public таблицу с информацией о флайтах в период от date_start до date_end
        """
        table_name = f"public.media_{channel.lower()}_results_trino"
        query_template = self.get_sql_query('media_results_trino_creator.sql')
        
        self.t_engine.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        query = query_template.format(
            table_name=table_name
        )
        
        self.t_engine.execute(query)
        
    def get_new_flights(self, channel, date_end): 
        table_name = f"public.media_{channel.lower()}_results_trino"
        
        flight_name = ''
        ps_flight_info = self.get_dataset_from_ps(channel, flight_name)
        ps_flight_info["tier"] = (
                ps_flight_info["tier"]               
                    .replace({"Tier 1": "T1",
                              "Tier 2": "T2",
                              "Tier 3": "T3"})
            )
        ps_flight_info['date_end'] = pd.to_datetime(ps_flight_info['date_end'])
        ps_flight_info = ps_flight_info[ps_flight_info['tier'].isin(['T1', 'T2', 'T3'])]
        ps_flight_info = ps_flight_info[(ps_flight_info['date_end'].dt.month == pd.to_datetime(date_end).month) 
                            & (ps_flight_info['date_end'].dt.year == pd.to_datetime(date_end).year)]
        
        flight_budget = self.get_flight_budget(flight_name)
        
        df = ps_flight_info.merge(flight_budget, how='inner', on='flight_name')
        final_df = pd.DataFrame({
            "flight_name": df["flight_name"],
            "vertical": df["vertical"],
            "category": df["category"],
            "type": df["adv_type"],
            "tier": df["tier"],
            "channel": channel,
            "date_start": df["date_start"],
            "date_end": df["date_end"],
            "analysed_date_start": df["date_start"],
            "analysed_date_end": df["date_end"],
            "flight_budget": df["flight_budget"],
            "channel_budget": df["budget"],
            "channel_budget_analysed": 0,
            "analysis_tool": None,
            "main_metric": None,
            "is_main_metric": None,
            "proxy_metric": None,
            "rel_metric_uplift": None,
            "rel_metric_mde": None,
            "abs_metric_uplift": None,
            "abs_metric_mde": None,
            "romi": None,
            "mde_romi": None,
            "revenue": None,
            "metric_to_revenue": None,
            "metric_to_revenue_elasticity": None,
            "cross_effect": None,
            "long_term_effect": None,
            "sum_metric_per_flight": None,
            "cost_per_metric": None,
            "verdict": None,
            "main_log_cats": None,
            "cf_link": None,
            "custom_slice_name_1": None,
            "custom_slice_uplift_1": None,
            "custom_slice_mde_1": None,
            "custom_slice_name_2": None,
            "custom_slice_uplift_2": None,
            "custom_slice_mde_2": None,
            "custom_slice_name_3": None,
            "custom_slice_uplift_3": None,
            "custom_slice_mde_3": None,
            "custom_slice_name_4": None,
            "custom_slice_uplift_4": None,
            "custom_slice_mde_4": None
        })
        
        return final_df


    def get_channel_table(self, channel):
        """
        Функция для получения информации о флайтах в выбранном канале
        :channel: TV, Digital, OOH или Other
        """
        table_name = f'public.media_{channel.lower()}_results_trino'
        
        query_template = self.get_sql_query('get_media_channel_results.sql')
        
        query = query_template.format(
            table_name=table_name
        )
        
        return self.t_engine.select(query)