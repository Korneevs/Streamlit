import sys
lib_dir = '/srv/data/my_shared_data/Lib'
if lib_dir not in sys.path:
    sys.path.append(lib_dir)
import re
import textwrap
from sql_formatter.core import format_sql
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import numpy as np
from tqdm.notebook import tqdm

from media.Validator.SliceDictParser import SliceDictParser
from media.Validator.UtilityDataFetcher import UtilityDataFetcher


class PeriodFinder():
    
    
    def __init__(self, engine_v, ps_table = 'public.ps_media_campaigns'):
        self.engine_v = engine_v
        self.ps_table = ps_table
        self.parser = SliceDictParser()
        self.util_data_fetcher = UtilityDataFetcher()
        self.reg_pop_dict = None
        
        
    def show_campaigns_in_slice(self, data_dict, sep='; '):
        """
        """
        query_dict = self.parser.parse_query_dict(data_dict, sep)
        verts = ', '.join([f"'{v}'" for v in query_dict['vertical']])
        if 'Any' in query_dict['logical_category']:
            log_cats_cond = "True"
        else:
            log_cats_cond = "REGEXP_LIKE(logical_category, '{regexp}')".format(regexp = '|'.join(['\\b' + re.escape(lc) + ';' for lc in query_dict['logical_category']]))
        channels = ', '.join([f"'{ch}'" for ch in query_dict['channel']])
        if 'Any' in query_dict['geo']:
            geo_cond = 'True'
        else:
            geo_cond = "geo IN ({geos})".format(geos = ', '.join([f"'{g}'" for g in query_dict['geo']]))
        dates = ', '.join([f"'{dt}'" for dt in query_dict['date']])
        
        
        query = format_sql(
        f"""
        SELECT vertical, rk_id, rk_name, logical_category, channel, geo, date, channel_daily_budget
        FROM {self.ps_table}
        WHERE vertical IN ({verts}) AND
        {log_cats_cond} AND
        channel IN ({channels}) AND
        {geo_cond} AND 
        date IN ({dates})
        """)
        
        res = self.engine_v.select(query)
        return res
    
    
    def was_campaign_in_slice(self, data_dict, sep = '; '):
        """
        """
        dfCampaigns = self.show_campaigns_in_slice(data_dict, sep)
        if len(dfCampaigns):
            return True
        else:
            return False
        
        
    def aggregate_campaigns(self, data_dicts, sep='; '):
        """
        """
        sorted_dicts = sorted(data_dicts, key=lambda x: x['weight'], reverse=True)
        slice_queries = []
        
        # Doesn't do anything, only for flexibility: channel_daily_budget should be already unique for rk_id, channel, date, geo
        total_budgets_query = format_sql(
        f"""
        WITH t_total_budgets AS (
        SELECT *, SUM(channel_daily_budget) OVER (PARTITION BY rk_id, channel, date, geo) as total_channel_daily_budget
        FROM {self.ps_table}
        )
        """)
        
        
        for priority, data_dict in enumerate(sorted_dicts):
            
            query_dict = self.parser.parse_query_dict(data_dict, sep)
            verts = ', '.join([f"'{v}'" for v in query_dict['vertical']])
            if 'Any' in query_dict['logical_category']:
                log_cats_cond = "True"
            else:
                log_cats_cond = "REGEXP_LIKE(logical_category, '{regexp}')".format(regexp = '|'.join(['\\b' + re.escape(lc) + ';' for lc in query_dict['logical_category']]))
            channels = ', '.join([f"'{ch}'" for ch in query_dict['channel']])
            
            assert len(query_dict['geo']) > 0
            if 'Any' in query_dict['geo']:
                geo_cond = 'True'
            else:
                geo_cond = "geo IN ({geos})".format(geos = ', '.join([f"'{g}'" for g in query_dict['geo']]))
            dates = ', '.join([f"'{dt}'" for dt in query_dict['date']])
            
            weight = data_dict['weight']
            
            query = format_sql(
            f"""
            SELECT vertical, rk_id, rk_name, logical_category, channel, geo, date,
            {weight} * total_channel_daily_budget as effective_daily_budget, {priority} as priority
            FROM t_total_budgets
            WHERE vertical IN ({verts}) AND
            {log_cats_cond} AND
            channel IN ({channels}) AND
            {geo_cond} AND 
            date IN ({dates})
            """)
            
            slice_queries.append(query)
        
        core_query = f'{total_budgets_query}\n' + 'UNION ALL\n'.join(slice_queries).strip()
        aggregation_query = format_sql(
        f"""
        WITH core AS (
        {core_query}
        )
        SELECT geo, date, sum(effective_daily_budget) as effective_daily_budget
        FROM 
            (
            SELECT *, min(priority) over (partition by rk_id, channel, geo, date) as min_priority
            FROM core
            ) core_min_priorities
        WHERE priority = min_priority
        GROUP BY 1, 2
        ORDER BY 1, 2
        """)
        res = self.engine_v.select(aggregation_query)
        return res
    
    
    def get_region_date_matrix(self, data_dicts, start, end, daily_budget_threshold=0, reference_region='РФ', sep='; '):
        """
        """
        interval = pd.date_range(start, end)
        df_camp_dates = self.aggregate_campaigns(data_dicts, sep)
        
        df_camp_dates = df_camp_dates.set_index(['geo', 'date'])\
            .reindex(pd.MultiIndex.from_product(
            (df_camp_dates['geo'].unique(), pd.date_range(start, end)), names = ['geo', 'date']
            ), fill_value = 0)\
            .reset_index()
        df_camp_dates = df_camp_dates\
            .pivot(index='geo', columns='date', values='effective_daily_budget')
        reg_threshold = self._calculate_regional_threshold(daily_budget_threshold, reference_region)
        df_camp_dates = df_camp_dates.loc[df_camp_dates.index.isin(reg_threshold), :]
        thresholds = df_camp_dates.index.map(lambda reg: reg_threshold[reg])
        for col in df_camp_dates.columns:
            df_camp_dates.loc[:, col] = df_camp_dates[col] * (df_camp_dates[col] >= thresholds)
        return df_camp_dates
    
    
    def visualize_region_date_matrix(self, data_dicts, start, end, daily_budget_threshold=0,
                                     reference_region='РФ', sep='; ', log_scale=False, save=False):
        df_camp_date = self.get_region_date_matrix(data_dicts, start, end, daily_budget_threshold, reference_region, sep)
        target_slice = self.parser.parse_query_dict(data_dicts[0], sep)
        vertical = '; '.join(target_slice['vertical'])
        log_cats = '; '.join(target_slice['logical_category'])
        channels = '; '.join(target_slice['channel'])
        z_title = "Эффективные подневные <br> затраты на все кампании, <br> идущие в регионе"
        expenses = df_camp_date.values
        if log_scale:
            df_camp_date = df_camp_date.apply(lambda x: np.log1p(x))
            z_title = "Эффективные подневные <br> затраты на все кампании, <br> идущие в регионе <br> (лог. масштаб)"
        title_text = f"Флайты в регионах для вертикали <b>{vertical}</b> логката/ов <b>{log_cats}</b> в каналах <b>{channels}</b>"
        title_text = '<br>'.join(textwrap.wrap(title_text, width=100))
        fig = px.imshow(df_camp_date, title=title_text, color_continuous_scale=px.colors.sequential.Reds)
        fig.update(data=[{'customdata': expenses,
            'hovertemplate': 'date: %{x}<br>geo: %{y}<br>effective expenses: %{customdata}<extra></extra>'}])
        fig.update_layout(coloraxis_colorbar_title_text=z_title, coloraxis_colorbar_title_font_size=8)
        fig.show()
        if save:
            fig.write_html(f"flight_periods.html")
    
    
    def get_clean_intervals(self, data_dicts, start, end, interval_length, prevalidation_clean_days=0, daily_budget_threshold=0,
                            reference_region='РФ', target_regs=None, excluded_intervals=None, sep='; '):
        """
        """
        rd_matrix = self.get_region_date_matrix(data_dicts, start, end, daily_budget_threshold, reference_region, sep)
        panels_list = []
        
        cur_date = rd_matrix.columns[-1]
        n_reg = len(rd_matrix)
        min_clean_ratio = 0.25

        while cur_date >= datetime.strptime(start, "%Y-%m-%d"):
            if cur_date - timedelta(prevalidation_clean_days + interval_length - 1) >= datetime.strptime(start, "%Y-%m-%d"):
                
                date_int_start = (cur_date - timedelta(interval_length - 1))
                prevalidation_start = date_int_start - timedelta(prevalidation_clean_days)
                
                reg_clean = (rd_matrix.loc[:, prevalidation_start:cur_date] == 0).all(axis=1)
                
                if target_regs is None or (set(target_regs) <= set(reg_clean.index)):
                    if reg_clean.sum() / n_reg >= min_clean_ratio:
                        if excluded_intervals is None or not self.check_not_excluded(prevalidation_start, 
                                                                                     cur_date, excluded_intervals):
                            panels_list.append((reg_clean[reg_clean].index, date_int_start, cur_date))
                            cur_date = date_int_start
                        
            cur_date -= timedelta(1)
        return panels_list
    
    
    def check_not_excluded(self, start, end, excluded_intervals):
        for left, right in excluded_intervals:
            left_date = datetime.strptime(left, '%Y-%m-%d')
            right_date = datetime.strptime(right, '%Y-%m-%d')
            
            if (left_date <= start <= right_date) or (start <= left_date <= end):
                return True
        return False
    
    
    def get_clean_intervals_yoy(self, data_dicts, start, end, yoy, interval_length, period_budget_diff_threshold, 
                                reference_region='РФ', excluded_intervals=None, sep='; ', **yoy_kwargs):
        
        rd_matrix = self.get_region_date_matrix(data_dicts, start, end, daily_budget_threshold=0,
                                                reference_region='РФ', sep=sep)
        
        learning_period = yoy_kwargs['learning_period']
        
        reg_marked_clean_days = []
        panels_list = []
        min_clean_ratio = 0.25
        
        for region in tqdm(rd_matrix.index):
        
            day_budgets = rd_matrix.loc[[region], :].sum(axis=0).to_dict()
            budget_df = pd.DataFrame.from_dict(day_budgets, orient='index').reset_index()
            budget_df.columns = ['ds', 'budget']

            budget_dict = dict()
            for start_ds in reversed(pd.date_range(
                datetime.strptime(start, '%Y-%m-%d') + timedelta(366 + 100) + timedelta(interval_length - 1), 
                datetime.strptime(end, '%Y-%m-%d') - timedelta(interval_length - 1), freq='D')):
                budget_dict[start_ds] = yoy.get_analysed_budget(budget_df, start_ds=start_ds, 
                                                                end_ds=start_ds + timedelta(interval_length - 1),
                                                                **yoy_kwargs)
                
            budget_diff = pd.DataFrame.from_dict(budget_dict, orient='index', columns=['analysed_budget'])
            cur_reg_threshold = period_budget_diff_threshold * self.reg_pop_dict[region] / self.reg_pop_dict[reference_region]
            budget_diff_marked = ((-cur_reg_threshold < budget_diff) & (budget_diff < cur_reg_threshold))
            
            for i, row in budget_diff_marked.iterrows():
                if row['analysed_budget'] == True:
                    budget_diff_marked.loc[(i - timedelta(1)):(i - timedelta(learning_period))] = False

            budget_diff_marked.columns = [region]
            reg_marked_clean_days.append(budget_diff_marked)
            
        
        clean_dates = pd.concat(reg_marked_clean_days, axis=1).transpose()
        num_all_regions = len(clean_dates.index)
        
        for date in clean_dates.columns:
            regs = clean_dates[date][clean_dates[date]].index
            if len(regs)/num_all_regions >= min_clean_ratio:
                if excluded_intervals is None or not self.check_not_excluded(date, 
                                                                             date + timedelta(interval_length - 1),
                                                                             excluded_intervals):
                    panels_list.append((regs, date, date + timedelta(interval_length - 1)))

        return panels_list 
        
    
    
    def _calculate_regional_threshold(self, daily_budget_threshold, reference_region):
        
        if self.reg_pop_dict is None:
            self.reg_pop_dict = self.util_data_fetcher.get_region_pops(self.engine_v)
        ref_reg_pop = self.reg_pop_dict[reference_region]
        reg_threshold = {reg: daily_budget_threshold * pop / ref_reg_pop for reg, pop in self.reg_pop_dict.items()}
        return reg_threshold
        