import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.stats as sps
from joblib import Parallel, delayed
from IPython.display import clear_output
from IPython.display import display, HTML
import os
from collections import defaultdict
import statsmodels.stats.api as sms
import plotly.express as px
from copy import copy, deepcopy
import seaborn as sns
from collections import namedtuple
from hashlib import blake2b
from tqdm.notebook import tqdm
from sklearn.metrics import mean_absolute_percentage_error
import sys
sys.path.append('../Lib')
import bxtools.vertica as vertica
import media.utils as utils

from media.TV_analyzer.SQL.final_platva_datasets import final_sql_dict
from media.TV_analyzer.stat_criterions import post_normed_ttest_for_full_effect


class PLATVA_results():

    def __init__(self, v_engine) -> None:
        """
            Инициализация класса
        """
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_columns', None)
        # Подключение к Vertica.
        self.vertica_engine = v_engine
        
        # Храним коэффициенты для каждой метрики и разреза.
        self.coefficients_dict = {}
    
    
    def get_coefficients(self):
        """
        Возвращает рассчитанные коэффициенты для каждой метрики и разреза.
        """
        
        return self.coefficients_dict
    
    
    def get_datasets_for_one_slice(self, name, final_dataset, 
                                   train_start_date, train_end_date,
                                   analysed_start_date, analysed_end_date,
                                   metrics, dc_result):
        
        # Рассчитываем коэффициенты для нормировки метрик относительно m42.
        coefficients_dict = {}

        for metric in metrics:
            # Числитель: сумма значений метрики из m42 за анализируемый период.
            num = dc_result['ml_datasets'][(metric, name)]['dataset'] \
                .query(f"date >= '{analysed_start_date}' and date <= '{analysed_end_date}'")['Any'].sum()

            # Знаменатель: сумма значений метрики за тот же период из витрины.
            den = self.vertica_engine.select(
                f"""
                SELECT SUM({metric})
                FROM {final_dataset}
                WHERE dt BETWEEN '{analysed_start_date}'::DATE AND '{analysed_end_date}'::DATE
                """
            ).values[0][0]
            
            # Рассчет коэффициента.
            if den != 0:
                coefficient = num / den
            else:
                raise ValueError(f"Нулевое значение метрики {metric} в аггрегации за аналитический период!")
                    
            coefficients_dict[metric] = coefficient
            
            # Сохраняем коэффициенты.
            self.coefficients_dict.setdefault(name, {})[metric] = coefficient
        
        date_metrics = ",\n".join([
            f"SUM({metric}) * {coefficients_dict[metric]} AS {metric}"
            for metric in metrics
        ])
        
        user_metrics = ",\n".join([
            f"""
            SUM(CASE WHEN dt <= '{train_end_date}'::DATE THEN {metric} ELSE 0 END) AS {metric}_before,
            SUM(CASE WHEN dt >= '{analysed_start_date}'::DATE THEN {metric} ELSE 0 END) AS {metric}_flight
            """ 
            for metric in metrics
        ])

        user_df = self.vertica_engine.select(
            final_sql_dict['user_group_df'].format(name=final_dataset, user_metrics=user_metrics)
        )
        
        other_user = self.vertica_engine.select(
            final_sql_dict['avito_sample_df'].format(name=final_dataset, user_metrics=user_metrics)
        )
        
        user_df = pd.concat([user_df, other_user])
        user_df = user_df.fillna(0).sort_values(by='user_id')
        
        date_df = self.vertica_engine.select(
            final_sql_dict['date_df'].format(name=final_dataset, date_metrics=date_metrics)
        )
        
        date_df = date_df.sort_values('dt')
        date_df = date_df.dropna()
        
        return user_df, date_df

    def analyse_one_slice(self, flight_name, slice_name, alpha, mediascope_reach,
                          final_dataset, train_start_date, train_end_date,
                          analysed_start_date, analysed_end_date, metrics, flight_budget, dc_result, **kwargs):
        
        user_df, date_df = self.get_datasets_for_one_slice(slice_name, final_dataset, 
                                   train_start_date, train_end_date,
                                   analysed_start_date, analysed_end_date,
                                   metrics, dc_result)
        
        show_results_df, debug_results_df = self._get_one_result(flight_name, slice_name, 
                                                    user_df, date_df, metrics, mediascope_reach, alpha, 
                                                    flight_budget,
                                                    train_start_date, train_end_date,
                                                    analysed_start_date, analysed_end_date)
    
        return show_results_df, debug_results_df
             
    def get_results(self, params, slice_table_dict, mediascope_reach, alpha, dc_result):
        show_dfs = []
        debug_dfs = []
        datasets = {}
        for slice_name, info_dict in tqdm(slice_table_dict.items()):
            params_copy = deepcopy(params)
            
            # Удалим в Json с параметрами флайта значения по этому ключу,
            # так как будем подтягивать теперь значение по этому ключу из slice_table_dict.
            del params_copy['metrics']
            
            show_results_df, debug_results_df = self.analyse_one_slice(slice_name=slice_name, alpha=alpha, 
                                                                       mediascope_reach=mediascope_reach, dc_result=dc_result,
                                                                       **params_copy, **info_dict)
            show_dfs.append(show_results_df)
            debug_dfs.append(debug_results_df)
            
        return pd.concat(show_dfs), pd.concat(debug_dfs)

    def _get_normed_test_sum(self, date_df, metric, analysed_start_date):
        # Берем условие на аналитический период.
        date_cond = (date_df['date'] >= analysed_start_date) 
        
        # Считаем сумму метрики за весь период по всему Авито.
        exp_period_sum = date_df[date_cond][metric].sum()

        return exp_period_sum
         
    def _normalyze_date_df(self, date_df_start, metric, mediascope_reach, train_end_date, analysed_start_date):
        date_df = date_df_start.copy()
        date_df['date'] = date_df['dt'].apply(lambda x: x.strftime('%Y-%m-%d'))

        # Условие для разделения на тренировочный и тестовый периоды.
        pre_cond = lambda df: df['date'] <= train_end_date
        after_cond = lambda df: df['date'] >= analysed_start_date
        
        # Нормализация метрики по среднему значению на тренировочном периоде.
        group_coeff = date_df[pre_cond(date_df)].groupby('exp_group')[metric].mean().to_dict()
        date_df['metric'] = date_df.apply(lambda row: row[metric] / group_coeff[row['exp_group']], axis=1)

        # Учитываем охват mediascope_reach.
        control_dict = date_df[date_df['exp_group'] == 'control'].set_index('date')['metric'].to_dict()
        date_df['metric'] = date_df.apply(
            lambda row: 
            (1 - mediascope_reach) * control_dict[row['date']] + 
            mediascope_reach * row['metric'] if row['exp_group'] == 'test' else row['metric'], 
            axis=1
        )

        # Деление на группы.
        test = date_df[date_df['exp_group'] == 'test'].sort_values('date')
        control = date_df[date_df['exp_group'] == 'control'].sort_values('date')

        # Суммирование для тестового периода у тестовой и контрольной группы.
        test_sum = test[after_cond(test)]['metric'].sum()
        control_sum = control[after_cond(control)]['metric'].sum()
        
        # Относительный эффект.
        rel_effect = test_sum / control_sum - 1
        
        # Суммарное значение метрики по всему Авито за весь аналитический период.
        exp_period_sum = self._get_normed_test_sum(date_df, metric, analysed_start_date)

        # Коррекция коэффициента с учетом смещения.
        # Нормируем таким образом, чтобы учесть базу метрики по всему Авито за весь аналитический период.
        coeff = rel_effect * exp_period_sum / (test_sum - control_sum)

        # Применение коэффициента к метрике в тесте и контроле.
        test['metric'] *= coeff
        control['metric'] *= coeff

        # Формирование результирующей таблицы.
        test = test[['date', 'metric']]
        test.columns = ['date', 'test_timeseries']
        
        control = control[['date', 'metric']]
        control.columns = ['date', 'control_timeseries']

        # Объединение данных в один DataFrame.
        df_new = pd.merge(control, test, on='date')

        return df_new

    def _add_cumsum(self, full_df, start_date):
        deltas = np.array(full_df['deltas'])
        before_size = full_df[full_df['flight_period'] == 0].shape[0]
        before_analyse_size = full_df[full_df['analysed_period'] == 0].shape[0]
        start_size = full_df[full_df['date'] < start_date].shape[0]
        predicts = np.array(full_df[f'control_timeseries'])
        abs_effect = []
        rel_effect = []
        prediction = []
        start_ind = 0
        for i in range(0, len(deltas)):
            if i == 0 or i == before_size - 1 or i == before_analyse_size - 1 or i == start_size - 1:
                start_ind = i + 1
                abs_effect.append(0)
                rel_effect.append(0)
                prediction.append(0)
                continue
            abs_effect.append(np.nansum(deltas[start_ind:(i + 1)]))
            rel_effect.append(np.nansum(deltas[start_ind:(i + 1)]) / np.nansum(predicts[start_ind:(i + 1)]))
            prediction.append(np.nansum(predicts[start_ind:(i + 1)]))
            
        return np.array(abs_effect), np.array(rel_effect), np.array(prediction)
 
    def _make_errors(self, df):
        if 'relative_error_upper' in df.columns:
            df['absolute_error_upper'] = df['control_cumsum'] * df['relative_error_upper']
            df['absolute_error_lower'] = df['control_cumsum'] * df['relative_error_lower']
        elif 'absolute_error' in df.columns:
            df['relative_error_upper'] = df['absolute_error_upper']  / (df['control_cumsum'] + 1e-5)
            df['relative_error_lower'] = df['absolute_error_lower'] / (df['control_cumsum'] + 1e-5)
            
        return df
    
    def _get_result_dataset(self, flight_name, slice_name, date_df, metric, mediascope_reach, alpha, results, train_start_date, 
                            train_end_date, analysed_start_date, analysed_end_date, flight_budget, **kwargs):
        
        final_df = self._normalyze_date_df(date_df, metric, mediascope_reach,
                                          train_end_date, analysed_start_date)
        
        final_df[f'relative_error_upper'] = results.ci_length * mediascope_reach / 2
        final_df[f'relative_error_lower'] = -results.ci_length * mediascope_reach / 2
        
        final_df['flight_period'] = (final_df['date'] >= analysed_start_date).astype(int)
        final_df['analysed_period'] = final_df['flight_period']
        
        final_df[f'relative_error_lower'] = final_df.apply(
            lambda x: x[f'relative_error_lower'] if x['analysed_period'] else 0, axis=1
        )
        final_df[f'relative_error_upper'] = final_df.apply(
            lambda x: x[f'relative_error_upper'] if x['analysed_period'] else 0, axis=1
        )
        final_df['deltas'] = final_df['test_timeseries'] - final_df['control_timeseries']
        
        abs_effect, rel_effect, control_cumsum = self._add_cumsum(final_df, train_start_date)
        
        final_df['abs. effect'] = abs_effect
        final_df['rel. effect'] = rel_effect
        final_df['control_cumsum'] = control_cumsum
        final_df = self._make_errors(final_df)

        final_df = final_df[final_df['date'] <= analysed_end_date]

        final_df['rel. CI left'] = final_df['rel. effect'] - final_df['relative_error_upper']
        final_df['rel. CI right'] = final_df['rel. effect'] - final_df['relative_error_lower']
        final_df['abs. CI left'] = final_df['abs. effect'] - final_df['absolute_error_upper']
        final_df['abs. CI right'] = final_df['abs. effect'] - final_df['absolute_error_lower']
        final_df['CI size'] = (final_df['abs. CI right']  - final_df['abs. CI left'])

        final_df = pd.DataFrame({
            'date': final_df['date'],
            'experiment analyzed money': flight_budget,
            'flight_period': final_df['flight_period'],
            'analysed_period': final_df['analysed_period'],
            'label': [flight_name] * len(final_df),
            'start_date': [analysed_start_date] * len(final_df),
            'analyse_start_date': [analysed_start_date] * len(final_df),
            'end_date': [analysed_end_date] * len(final_df),
            'flight_end_date': [analysed_end_date] * len(final_df),
            'slice_name': [slice_name] * len(final_df),
            'control_features': [['Искусственный контроль, основанный на несмотрящих ТВ пользователях']] * len(final_df),
            'test_locations': [['Все пользователи']] * len(final_df),
            'debug': False,
            'alpha': alpha,
            'metric': metric,
            'test_timeseries': final_df['test_timeseries'],
            'control_timeseries': final_df['control_timeseries'],
            'period': final_df['flight_period'].apply(lambda x: 'before' if x == 0 else 'after'),
            'diff': final_df[f'deltas'],
            'abs. effect': final_df['abs. effect'],
            'rel. effect': final_df['rel. effect'],
            "rel. lower_error_bound": final_df['relative_error_lower'],
            "rel. upper_error_bound": final_df['relative_error_upper'],
            "abs. lower_error_bound": final_df['absolute_error_lower'],
            "abs. upper_error_bound": final_df['absolute_error_upper'],
            "rel. CI left": final_df['rel. CI left'],
            "rel. CI right": final_df["rel. CI right"],
            "abs. CI left": final_df["abs. CI left"],
            "abs. CI right": final_df["abs. CI right"],
            "CI size": final_df['CI size'],
            'method': ["PLATVA"] * len(final_df),
        })
        
        fin_effect = final_df[final_df['date'] <= analysed_end_date].iloc[-1]['rel. effect']

        min_date = final_df[final_df['date'] >= analysed_start_date]['date'].min()
        final_df.loc[final_df['date'] == min_date, 'period'] = 'after'
        
        return final_df
    
    def _get_one_result(self, flight_name, slice_name, 
                       df, date_df, metrics, mediascope_reach, alpha, 
                       flight_budget,
                        train_start_date, train_end_date,
                        analysed_start_date, analysed_end_date, **kwargs):
        row_raw_results = []
        results_dfs = []
        for metric in metrics:
            params = {}
            for name in ['test', 'control']:
                curr = df[df['exp_group'] == name]
                params[f'after_{name}'] = np.array(curr[f'{metric}_flight'])
                params[f'before_{name}'] = np.array(curr[f'{metric}_before'])
                
            full_results = post_normed_ttest_for_full_effect(
                alpha=alpha, show_tv_group_results=False, **params)
            
            tv_watchers_uplift_results = post_normed_ttest_for_full_effect(
                alpha=alpha, show_tv_group_results=True, **params)

            row = {
                'slice_name': slice_name,
                'metric': metric,
                'alpha': alpha,
                'effect': full_results.effect * mediascope_reach,
                'left_bound': full_results.left_bound * mediascope_reach,
                'right_bound': full_results.right_bound * mediascope_reach,
                'MDE': full_results.ci_length * mediascope_reach / 2,
                'TV users uplift': tv_watchers_uplift_results.effect,
                'TV left_bound': tv_watchers_uplift_results.left_bound,
                'TV right_bound': tv_watchers_uplift_results.right_bound,
                'TV MDE': tv_watchers_uplift_results.ci_length / 2,
                'train_start_date': train_start_date,
                'train_end_date': train_end_date,
                'analysed_start_date': analysed_start_date,
                'analysed_end_date': analysed_end_date
            }
            row_raw_results.append(row)

            show_results_df = self._get_result_dataset(flight_name, slice_name, date_df, metric, mediascope_reach, alpha, 
                                                       tv_watchers_uplift_results, train_start_date, train_end_date, 
                                                       analysed_start_date, analysed_end_date, flight_budget, **kwargs)
            results_dfs.append(show_results_df)
            
            title = f"{flight_name}: {slice_name}, {metric}"
            self.check_results_for_validity(title, df, date_df, metric, train_end_date, analysed_start_date, alpha)
            
        raw_results_df = pd.DataFrame(row_raw_results)
        show_results_df = pd.concat(results_dfs)
        
        return show_results_df, raw_results_df
    
    def show_group_dynamics(self, title, date_df_start, train_end_date, analysed_start_date, metric):
        date_df = date_df_start.copy()
        date_df = date_df.sort_values('dt')
        
        # Условия на тренировочный и аналитический период.
        pre_cond = lambda df: df['dt'] <= pd.to_datetime(train_end_date)
        after_cond = lambda df: df['dt'] >= pd.to_datetime(analysed_start_date)
        
        # Вычисляем среднее значение метрики за предпериод.
        group_coeff = date_df[pre_cond(date_df)].groupby('exp_group')[metric].mean().to_dict()
        
        # Нормируем метрику на среднее на предпериоде, которое вычислили выше.
        date_df[metric] = date_df.apply(lambda row: row[metric] / group_coeff[row['exp_group']], axis=1)
        
        # Создаем три группы: Тест / Контроль / Авито.
        test = date_df[date_df['exp_group'] == 'test']
        control = date_df[date_df['exp_group'] == 'control']
        avito = date_df[date_df['exp_group'] == 'Avito']
        
        # Вычисляем относительный эффект.
        effect = test[after_cond(test)][metric].sum() / control[after_cond(control)][metric].sum() - 1
        print(f"Относительный эффект за период анализа: {round(effect * 100, 1)}%")
        
        # Вычисляем MAPE между Авито пользователями и тестовыми пользователями (те, кто посмотрели рекламу).
        mape = mean_absolute_percentage_error(avito[after_cond(avito)][metric], test[after_cond(test)][metric])
        print(f"MAPE(Avito, TV users): {round(mape * 100, 1)}%")
        
        # Рисуем график.
        fig = px.line(date_df, x="dt", y=metric, color='exp_group', 
                      color_discrete_map={'test':'red', 'control':'blue', 'Avito':'grey'})
        
        fig.update_layout(
            title = title,
            title_font = {"size": 20, "family": "Arial"},
        )
        
        # Выводим график.
        fig.show()

    def check_results_for_one_group_vs_AVITO(self, user_df, metric, group, correct_sign, alpha):
        params = {}
        for name, save_name in zip(['Avito', group], ['control', 'test']):
            curr = user_df[user_df['exp_group'] == name]
            params[f'after_{save_name}'] = np.array(curr[f'{metric}_flight'])
            params[f'before_{save_name}'] = np.array(curr[f'{metric}_before'])
        results = post_normed_ttest_for_full_effect(alpha=alpha, show_tv_group_results=True, **params)
        if correct_sign == '+':
            bad = results.right_bound < 0
        else:
            bad = results.left_bound > 0
        alert = '' #if bad == 0 else 'ALERT!'
        print(f"Effect {group} minus Avito: ({round(results.left_bound * 100, 1)}%,"\
              f" {round(results.right_bound * 100, 1)}%) {alert}; MDE: {round(results.ci_length * 100 / 2, 1)}%")
        print(f"================")

    def check_results_for_validity(self, title, user_df, date_df, metric, train_end_date, analysed_start_date, alpha):
        print(title)
        self.show_group_dynamics(title, date_df, train_end_date, analysed_start_date, metric)
      