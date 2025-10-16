import sys
sys.path.append('../Lib')


import getpass
import pandas as pd
import numpy as np
import requests
import warnings
import datetime
import re
import string
import warnings
from collections import defaultdict
from tqdm.notebook import tqdm
import plotly.graph_objects as go
from copy import deepcopy
from plotly.subplots import make_subplots
import re
from media.Visualizer.common import format_with_suffix, make_link
import media.Visualizer.colors as colors


def beauty_iteration(slice_full):
    for slice_name in slice_full:
            yield slice_name.split('@@@')

class ShowingBeauty():
    
    def __init__(self, engine) -> None:
        """
            Инициализация класса
        """
        self.ch_engine = engine
        self.metrics_dict = engine.select("select * FROM dct.m42_metric").set_index('metric')['metric_id'].to_dict()
    

    def make_one_plot(self, fig, data, visible, metric):
        ##
        fig.add_trace(
            go.Scatter(
                line=dict(color=colors.grey, width=2.5),
                name='CI',
                x=data['date'],
                y=data['abs. upper_error_bound'].replace(0, np.nan),
                mode='lines',
                hoverlabel=dict(align="left"),
                hovertemplate='%{x}<br>%{text}',
                text=f"{metric}:<br>abs. upper bound: "\
                    + data['abs. upper_error_bound'].apply(lambda x: format_with_suffix(x, signed=True))
                    +"<br>rel. upper bound: " +\
                    + data['rel. upper_error_bound'].apply(
                            lambda x: format_with_suffix(x * 100, signed=True, percent=True)
                      ).astype(str),
                visible=visible,
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                line=dict(color=colors.grey, width=2.5),
                name='CI',
                x=data['date'],
                y=data['abs. lower_error_bound'].replace(0, np.nan),
                mode='lines',
                fill='tonexty',
                hoverlabel=dict(align="left"),
                hovertemplate='%{x}<br>%{text}',
                text=f"{metric}:<br>abs. lower bound: "\
                    + data['abs. lower_error_bound'].apply(lambda x: format_with_suffix(x, signed=True))
                    +"<br>rel. lower bound: " +\
                    + data['rel. lower_error_bound'].apply(
                            lambda x: format_with_suffix(x * 100, signed=True, percent=True)
                      ).astype(str),
                visible=visible,
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                line=dict(color=colors.green, width=2.5),
                name='Cumulative Uplift',
                x=data['date'],
                y=data['abs. effect'],
                mode='lines',
                hoverlabel=dict(align="left"),
                hovertemplate='%{x}<br>%{text}',
                text=f"{metric}:<br>abs. cumulative effect: "\
                    + data['abs. effect'].apply(lambda x: format_with_suffix(x, signed=True))
                    +"<br>rel. cumulative effect: " +\
                    + data['rel. effect'].apply(
                            lambda x: format_with_suffix(x * 100, signed=True, percent=True)
                      ).astype(str),
                visible=visible,
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
        ##
        fig.add_scatter(
            line=dict(color=colors.red, width=2.5),
            name='Test',
            x=data['date'],
            y=data['test_timeseries'],
            row=2,
            col=1,
            mode='lines',
            hovertemplate='%{x}<br>%{text}',
            text=f"{metric}: "+ data['test_timeseries'].apply(lambda x: format_with_suffix(x, signed=False)) + "",
            visible=visible,
        )
        fig.add_scatter(
            line=dict(color=colors.blue, width=2.5),
            name='Control',
            x=data['date'],
            y=data['control_timeseries'],
            row=2,
            col=1,
            mode='lines',
            hovertemplate='%{x}<br>%{text}',
            text=f"{metric}: "+ data['control_timeseries'].apply(lambda x: format_with_suffix(x, signed=False)) + "",
            visible=visible,
        )
    
        fig.add_bar(
            marker=dict(color=colors.green),
            name='Uplift',
            x=data['date'],
            y=data['diff'],
            row=3,
            col=1,
            hovertemplate='%{x}<br>%{text}',
            text=f"{metric} delta: "+ data['diff'].apply(lambda x: format_with_suffix(x)) + "",
            visible=visible,
        )
        return fig
        
    
    def _init_fig(self):
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=['Cumulative Uplift', 'Metric Dynamic', 'Uplift'],
            specs=[
                [{'secondary_y': True}], 
                [{'secondary_y': False}],
                [{'secondary_y': False}]
            ],
        )
        fig.update_layout(
            template=colors.big10_theme,
            height=300 * 3 + 400,
            showlegend=False,
            legend_tracegroupgap=0,
            margin=dict(l=40, r=30, b=400, t=25, pad=0, autoexpand=True),
        )
        for i in fig['layout']['annotations']:
            i['font'] = dict(size=14, color='black')
        return fig
    
    
    def add_vline(self, title, pos, date):
        return [dict(
            type='line',
            x0=datetime.datetime(date.year, date.month, date.day).timestamp() * 1000,
            x1=datetime.datetime(date.year, date.month, date.day).timestamp() * 1000,
            xref=f'x{i}' if i != 1 else f'x',
            y0=0,
            y1=1,
            yref=f'y{i + 1} domain' if i != 1 else f'y domain',
            line={
                'dash': 'dash',
                'width': 3,
                'color': colors.grey,
            },
        ) for i in [1, 2, 3]
       ]


    def add_vline_annotation(self, title, xanchor, date):
        return [dict(
             font={'color': colors.grey, 'size': 10},
             showarrow=False,
             text=title,
             x=datetime.datetime(date.year, date.month, date.day).timestamp() * 1000,
             xanchor=xanchor,
             xref=f'x{i}' if i != 1 else f'x',
             y=1,
             yanchor='top',
             yref=f'y{i + 1} domain' if i != 1 else f'y domain',
        ) for i in [1, 2, 3]
        ]
        
   
    def _add_exp_dates_vlines(self, start_date, end_date, flight_end_date):
        array = [] 
        array += self.add_vline('Start', 'top left', start_date)
        array += self.add_vline('Analyse period end', 'top right', end_date)
        array += self.add_vline('Flight end', 'top left', flight_end_date)
        return array
    
    
    def _add_exp_dates_vline_annotations(self, start_date, end_date, flight_end_date):
        array = [] 
        array += self.add_vline_annotation('Start', 'right', start_date)
        array += self.add_vline_annotation('Analyse period end', 'left', end_date)
        array += self.add_vline_annotation('Flight end', 'right', flight_end_date)
        return array

    
    def _show_appendix_text(self, array, start_size):
        result = ""
        size = start_size
        for elem in array:
            split = ", "
            if size > 100:
                size = 0
                split = ", <br>"
            result += split + elem
            size += len(elem) + 2
        return result[2:]
    
    
    def _make_timeseries_html_plots(self, fig, grouped_df, metrics, slices, slices_json):
        index_i_corr = {}
        full_to_show=[]
        annotations={}
        shapes = {}
        FIGS_IN_ONE=6
        size = len(grouped_df.index.unique())
        slice_df = grouped_df.reset_index()
        slice_full = (slice_df['slice_name'] + '@@@' + slice_df['metric']).unique()
        for i, (slice_name, metric) in enumerate(beauty_iteration(slice_full)):
            key = (slice_name, metric)
            index_i_corr[key] = i
            curr_df = grouped_df.loc[key]
            to_show = [False for _ in range(FIGS_IN_ONE * size)] #+ [True, True]
            for j in range(FIGS_IN_ONE * i, FIGS_IN_ONE * (i + 1)):
                to_show[j] = True
            full_to_show.append(to_show)
            
            
            control = "<b>Фичи</b>: " + \
                    self._show_appendix_text(list(curr_df['control_features'].iloc[0]), start_size=len('Фичи'))
            test = "<b>Тест</b>: " + self._show_appendix_text(list(curr_df['test_locations'].iloc[0]), start_size=len('Тест'))
            appendix = test + "<br><br>" + control
            visible = False
            if i == 0:
                visible = True
            fig = self.make_one_plot(fig, curr_df, visible, metric)
            text = ""
            if slice_name in slices_json:
                try:
                    text = make_link(slices_json[slice_name]['target'], metric, self.metrics_dict)
                except Exception:
                    text = ""
                    
            start_date = curr_df.iloc[0]['start_date']
            end_date = curr_df.iloc[0]['end_date']
            flight_end_date = curr_df.iloc[0]['flight_end_date']
            
            shapes[(slice_name, metric)] = self._add_exp_dates_vlines(start_date, end_date, flight_end_date)
            annotations[(slice_name, metric)] = [
                go.Annotation(
                    text=text,
                    x=0,
                    y=1.09,
                    xanchor="left",
                    yanchor="top",
                    xref='paper',
                    yref='paper',
                    align='left',
                    showarrow=False,
                    visible=True,
                    font=dict(color='black', size=24, family='Times New Roman')
                ),
                go.Annotation(
                    text=appendix,
                    x=0,
                    y=-0.12,
                    xanchor="left",
                    yanchor="top",
                    xref='paper',
                    yref='paper',
                    align='left',
                    showarrow=False,
                    visible=True,
                    font=dict(color='black', size=15, family='Times New Roman')
                ),
            ] + self._add_exp_dates_vline_annotations(start_date, end_date, flight_end_date)
            if i == 0:
                save_to_add_shape = shapes[(slice_name, metric)]
                save_to_add_ann = annotations[(slice_name, metric)]
        
        start_ann = deepcopy(list(fig['layout']['annotations']))
        start_shape = deepcopy(list(fig['layout']['shapes']))
        fig['layout']['shapes'] = list(fig['layout']['shapes']) + save_to_add_shape
        fig['layout']['annotations'] = list(fig['layout']['annotations']) + save_to_add_ann

        fig.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    active=0,
                    showactive=True,
                    xanchor="left",
                    yanchor="top",
                    x=0, y=1.16,
                    buttons=list([
                        dict(label=f"slice: {slice_name}, metric: {metric}",
                             method="update",
                             args=[{"visible": full_to_show[index_i_corr[(slice_name, metric)]]},
                                   {"annotations": start_ann + annotations[(slice_name, metric)],
                                    "shapes": start_shape + shapes[(slice_name, metric)]
                                   },
                                  ]
                            ) for slice_name, metric in beauty_iteration(slice_full)
                    ])
                ),
            ]
        )
        return fig

    def make_timeseries_result(self, df, slices_json, debug=False):
        grouped_df = df.set_index(['slice_name', 'metric'])
        if not debug:
            grouped_df = grouped_df[grouped_df['debug'] == False]
        
#         if debug:
#             new_dates = pd.date_range(start='2020-01-01', end='2022-01-01')
#             dates_dict = {date: new_date.strftime('%Y-%m-%d')
#                           for date, new_date in zip(sorted(grouped_df['date'].unique()), new_dates)}
#             grouped_df['date'] = grouped_df['date'].apply(lambda date: dates_dict[date])
#             grouped_df['start_date'] = grouped_df['start_date'].apply(lambda date: dates_dict[date])
#             grouped_df['end_date'] = grouped_df['end_date'].apply(lambda date: dates_dict[date])
#             grouped_df['flight_end_date'] = grouped_df['flight_end_date'].apply(lambda date: dates_dict[date])


        grouped_df['date']       = pd.to_datetime(grouped_df['date'])
        grouped_df['start_date'] = pd.to_datetime(grouped_df['start_date'])
        grouped_df['end_date']   = pd.to_datetime(grouped_df['end_date'])
        grouped_df['flight_end_date']   = pd.to_datetime(grouped_df['flight_end_date'])
        
        fig = self._init_fig()
        fig = self._make_timeseries_html_plots(fig, grouped_df, df['metric'].unique(), df['slice_name'].unique(), 
                                               slices_json)
#         print(fig.data)
        return fig


    def show_flight_results(self, df, slices_json, negative_metrics=[], rev_per_metric=None, save=False):
        tr = TableResults()
        title = df['label'].iloc[0]
        fig_table = tr.get_show_tables(df, negative_metrics, rev_per_metric)
        fig_ts = self.make_timeseries_result(df, slices_json, debug=False)
        if save:
            fig_table.write_image(f"{title}_table.png", scale=2)
            fig_ts.write_html(f'{title}_timeseries.html')
        fig_table.show()
        fig_ts.show()
        return 
        
        
    def show_errors(self, df, slice_name, metric, save=False, limit=10):
        curr_df = df[(df['metric'] == metric) &\
                     (df['slice_name'].apply(lambda x: re.match(f'.*: {slice_name}', x) is not None))]
        errors = []
        for _, dataset in curr_df.groupby(['slice_name', 'metric']):
            slice_name = dataset['slice_name'].iloc[0]
            start_date = dataset['start_date'].iloc[0]
            end_date = dataset['end_date'].iloc[0]
            effect = dataset[dataset['date'] == end_date].iloc[-1]['rel. effect'] * 100
            budg = -1
            if 'experiment analyzed money' in dataset.columns:
                budg = dataset[dataset['date'] == end_date].iloc[-1]['experiment analyzed money']
            errors.append((slice_name, effect, start_date, end_date, budg))


        errors = sorted(errors, key=lambda x: -abs(x[1]))
        
        if limit != -1:
            errors = errors[:limit]
        show_names = []
        for slice_name, effect, start_date, end_date, budg in errors:
            show_names.append(slice_name)
            effect = format_with_suffix(effect, signed=True, percent=True)
            if budg == -1:
                print(f'{slice_name}, period: {start_date} - {end_date}; effect: {effect}')
            else:
                budg = format_with_suffix(budg, signed=True, percent=False)
                print(f'{slice_name}, period: {start_date} - {end_date}, budget: {budg}; effect: {effect}')
        
        curr_df = curr_df[curr_df['slice_name'].isin(show_names)].sort_values(['date', 'slice_name'])
        fig_ts = self.make_timeseries_result(curr_df, {}, debug=False)
        return fig_ts

        
        


class TableResults():

    def __init__(self) -> None:
        """
            Инициализация класса
        """
        warnings.filterwarnings("ignore")
        self.vertical_size = 10
        self.baseline_size = 11
        self.no_effect_color = '#C0C0C0'
        self.green_text_color = '#00AA00'
        self.red_text_color = '#CD212A'
        
        self.background_green_color = '#90ED90'
        self.background_red_color = '#FF6F61'
        
        
        self.background_color_to_use = {
            -1: self.background_red_color,
            1: self.background_green_color
        }
        
        self.text_color_to_use = {
            -1: self.red_text_color,
            1: self.green_text_color,
            0: self.no_effect_color
        }
    
    
    def make_effect(self, row, eff_type, coeff = 0):
        if eff_type == 'relative':
            upp = format_with_suffix(row[f'rel. effect'] * 100, signed=True, percent=True)
            ci_left = format_with_suffix(row[f'rel. CI left'] * 100, signed=True, percent=True)
            ci_right = format_with_suffix(row[f'rel. CI right'] * 100, signed=True, percent=True)
        elif eff_type == 'absolute':
            upp = format_with_suffix(row[f'abs. effect'], signed=True, percent=False)
            ci_left = format_with_suffix(row[f'abs. CI left'], signed=True, percent=False)
            ci_right = format_with_suffix(row[f'abs. CI right'], signed=True, percent=False)
        else:
            upp = format_with_suffix(row[f'abs. effect'] * coeff * 100, signed=False, percent=True)
            ci_left = format_with_suffix(row[f'abs. CI left'] * coeff * 100, signed=False, percent=True)
            ci_right = format_with_suffix(row[f'abs. CI right'] * coeff * 100, signed=False, percent=True)
        
        ci = f"[{ci_left}, {ci_right}]"
#         adder = 1 if (len(ci) - len(upp)) % 2 == 1 else 0
        effect = " " * (int((len(ci) - len(upp)) // 2 * 1.5) + 1) + \
                    upp + '<br>' +\
        "-" * int(len(ci) * 1.5 - 1) + '<br>' + ci
        return effect
    
    
    def get_rub_weight(self, metric, rev_per_metric, row):
        if rev_per_metric is None or metric not in rev_per_metric:
            if 'metric_weight' in row:
                return row['metric_weight']
            return 0
        return rev_per_metric[metric]
    
    
    def get_raw_dataframe(self, ts_df, negative_metrics, rev_per_metric):
        ts_df = ts_df.sort_values('date')
#         df = ts_df[ts_df['date'] <= ts_df['end_date']].sort_values('date').iloc[-1]
        rows = []

        slices = ts_df['slice_name'].unique()
        metrics = ts_df['metric'].unique()
        
        slice_full = sorted((ts_df['slice_name'] + '@@@' + ts_df['metric']).unique())
        
        
        for slice_name, metric in beauty_iteration(slice_full):
#             print(ts_df[(ts_df['metric'] == metric) & (ts_df['slice_name'] == slice_name) &\
#                                  (ts_df['date'] <= ts_df['end_date'])])
            row = ts_df[(ts_df['metric'] == metric) & (ts_df['slice_name'] == slice_name) &\
                                 (ts_df['date'] <= ts_df['end_date'])].iloc[-1]
            budg = row['experiment analyzed money']

#             assert len(curr) == 1, f"{metric} and {slice_name} not found!"
            
            bound = 'upper'
            alternative = 'greater'
            if row['metric'] in negative_metrics:
                bound = 'lower'
                alternative = 'less'
                
            sign = 0
            coeff = 1
            if alternative == 'less':
                coeff == -1
            if row[f'rel. effect'] > row[f'rel. upper_error_bound']:
                sign = coeff
            elif row[f'rel. effect'] < row[f'rel. lower_error_bound']:
                sign = -1 * coeff
            alpha = row['alpha'] / 2 * 100
            alpha = "{:.1f}".format(alpha)
            mde_space = ' ' * 2
            
            rub_weight = self.get_rub_weight(metric, rev_per_metric, row)
            roi = 'pass'
            if rub_weight is not None:
                roi = self.make_effect(row, eff_type='ROI', coeff= rub_weight / budg)
            row = {
                'slice': str(row['slice_name']),
                f"analysed<br>  budget": format_with_suffix(budg),
                'metric': row['metric'],
                "alternative": alternative,
                "rel. effect": self.make_effect(row, eff_type='relative'),
                'abs. effect': self.make_effect(row, eff_type='absolute'),
                #format_with_suffix(row[f'rel. effect'] * 100, signed=True, percent=True),
                f' rel. MDE<br>(α={alpha}%)': format_with_suffix(row[f'rel. {bound}_error_bound'] * 100, 
                                                                 signed=True, percent=True),
#                 'abs. effect': self.make_effect(row, eff_type='absolute'),
#                 f'{mde_space} abs. MDE<br>(alpha={alpha}%)': format_with_suffix(row[f'abs. {bound}_error_bound'],
#                                                                  signed=True, percent=False),
                'ROI': roi,
                "MDE ROI": 'pass' if rub_weight is None else format_with_suffix(
                    row[f'abs. {bound}_error_bound'] * rub_weight / budg * 100, signed=False, percent=True
                ),
                "sign": sign,
            }
            rows.append(row)
        return pd.DataFrame(rows)
    
    
    def _get_column_fill_text_colors(self, column, signs, start_fill_colors, start_text_colors):
        background = []
        text = []
        if 'effect' in column or 'MDE' in column or 'ROI' in column:
            for sign, default_color, text_color in zip(signs, start_fill_colors, start_text_colors):
                if sign == 1:
                    if 'MDE' in column:
                        background.append(self.background_green_color)
                        text.append(text_color)
                    else:
                        background.append(default_color)
                        text.append(self.green_text_color)
                elif sign == -1:
                    if 'MDE' in column:
                        background.append(self.background_red_color)
                        text.append('white')
                    else:
                        background.append(default_color)
                        text.append(self.red_text_color)
                else:
                    if 'MDE' in column:
                        background.append(default_color)
                        text.append(text_color)
                    else:
                        background.append(default_color)
                        text.append(self.no_effect_color)
        else:
            for default_color, text_color in zip(start_fill_colors, start_text_colors):
                background.append(default_color)
                text.append(text_color)
        return {
            'backround': background,
            'text': text
        }
   
    def _add_br_in_text(self, text):
        size = 15
        res = []
        prev_sz = 0

        for t in text.split('<br>'):
            for i in range(0, len(t), size):
                right = min(len(t), i + size)
                res.append(t[i:right] + "<br>")
        return "".join(res)


    def _get_text_height(self, df_col):
        start_size = len(str(list(df_col)).split('<br>'))
        size = start_size + len(str(list(df_col)).split("<br>',")) * 1.5
        return size


    def _get_html_table(self, df):
        signs = df['sign']
        df['slice']  = df['slice'].apply(lambda x: self._add_br_in_text(x))
        df['metric'] = df['metric'].apply(lambda x: self._add_br_in_text(x))
        text_rows_size = max(self._get_text_height(df['metric']), 10)
        text_rows_size = max(self._get_text_height(df['slice']), text_rows_size) + 2
        columns = df.columns[:-1]
        
        rowEvenColor = '#f5f5f5'
        rowOddColor = 'white'

        start_fill_colors = [rowOddColor if i % 2 == 0 else rowEvenColor for i in range(len(df))]
        start_text_colors = ['#2a3f5f' for i in range(len(df))]
        column_title_size = self.baseline_size + 1
        sizes = [self.baseline_size for column in columns]
        header=dict(values=columns, 
                    font=dict(family="Arial", size = column_title_size, color='black'), 
                    fill_color='lightgray', line_color='lightgray',
#                     height=header_hight,
                    align = 'center'
                   )
        values = [df[column] for column in columns]
        family = ['Arial' for column in columns]
        fill_colors = [self._get_column_fill_text_colors(column, signs, 
                             start_fill_colors, start_text_colors)['backround']
                       for ind, column in enumerate(columns)]
        text_colors = [self._get_column_fill_text_colors(column, signs, 
                             start_fill_colors, start_text_colors)['text']
                       for ind, column in enumerate(columns)]
        
        return text_rows_size, go.Table(
                header=header,
                cells=dict(values=values,
#                          height= self.current_size * 3, 
                           font=dict(family=family, size=sizes, color=text_colors),
                           fill_color=fill_colors,
                           line_color=fill_colors
                          ),
             )

    
    
    def get_show_tables(self, ts_df, negative_metrics=[], rev_per_metric=None):
        df = self.get_raw_dataframe(ts_df, negative_metrics, rev_per_metric)
    
        fig = make_subplots(
            rows=1,
            cols=1,
            vertical_spacing=0.04,
            horizontal_spacing = 0.01,
            subplot_titles=[f"Результаты"],
            specs=[
                [{"type": "table"}],
            ],
        )
        title = ts_df['label'].iloc[0]
        top = 80 if '<br>' in title else 80
        height, table = self._get_html_table(df)
        
        fig.update_layout(
            title = title,
            title_font = {"size": 20, "family": "Arial"},
            template=colors.big10_theme,
            height=height * (self.baseline_size + 20),
            width=900,
            showlegend=False,
            legend_tracegroupgap=0,
            margin=dict(l=0, r=0, b=25, t=top, pad=0, autoexpand=True),
        )
        fig.add_trace(
            table,
            row=1,
            col=1,
        )
        
        return fig
