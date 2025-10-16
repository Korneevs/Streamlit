from TableParser import GoogleSheetsParser
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

import plotly.express as px
import sys

lib_dir = '/Users/asekorneev/Downloads/Lib'
if lib_dir not in sys.path:
    sys.path.append(lib_dir)
from media.DatasetWorker.DatasetCreator import DatasetCreator
import numpy as np
import plotly as plt
import pickle
import math
from statsmodels.tsa.api import SimpleExpSmoothing
import datetime
import sys
import re
from tqdm.notebook import tqdm
import numpy as np
from password import password
from datetime import date


class pROMIser_t:
    def __init__(self,
                 c_engine,
                 t_engine,
                 question='Основная идея рекламы_Спонтанно_Авито',
                 real_seen_coeff=0.6,
                 min_sov=0.13,
                 max_sov=0.18,
                 sov_coeff_start=1.0,
                 sov_coeff_end=1.3,
                 start_rk='default',
                 campaigns_path='1nymqZ55uunXRlnNakDjRx6WPkbd-YBc4O9nFqolqzjU',
                 metrics_path='1jVFhIIPQ0ZJZMijbOcyBV6o7qJG9TQB_pUepgP3BM5A',
                 campaigns_list=[],
                 flight_trp_dict = None,
                 df = None,
                 data_to_predict = None,
                 cons=0,
                 opm=0):
        

        self.real_seen_coeff = real_seen_coeff
        self.min_sov = min_sov
        self.max_sov = max_sov
        self.sov_coeff_start = sov_coeff_start
        self.sov_coeff_end = sov_coeff_end
        self.question = question
        self.flight_data = {} 
        self.flight_trp_dict = {} if flight_trp_dict is None else flight_trp_dict
        self.start_rk = start_rk
        self.seen_curves = None
        self.df = df
        self.data_to_predict = data_to_predict
        token_path = "/Users/asekorneev/Documents/Work projects/Доки для кода/token.json"
        credentials_path = "/Users/asekorneev/Documents/Work projects/Доки для кода/client_secret_298388933431-9i3edv4spo4r4qro3cdo8l9u891hvgrp.apps.googleusercontent.com.json"

        self.parser = GoogleSheetsParser(token_path=token_path,
                            credentials_path=credentials_path)
        
        self.start_TRP = None
        self.end_TRP = None
        
        self.campaigns_path = campaigns_path
        self.metrics_path = metrics_path
        self.campaigns_list = campaigns_list
        self.cons = cons
        self.opm = opm

        self.closest_start_rk = {
                  'CC-Test': [
                    'Avito_Goods_gc-24-01-02-TRX-Buyer-Protection-T1-C2C',
                    'Avito_Goods_gc-24-09-AvitoMall+T&S-T1'
                  ],
                  'HL-Construction': [
                    'Avito_Goods_gc-24-06-H&L-T1-Construction-SALE',
                    'Avito_Goods_gc-24-04-HL-Construction-XY-T1-C2C',
                    'Avito_Goods_gc-25-04-H&L-T1-Construction'
                  ],
                  'HL-Furniture': [
                    'Avito_Goods_gc-24-10-HL-Furniture-Feature-C2C'
                  ],
                  'SP': [
                    'Avito_Goods_gc-24-05-SP-RoadTrips-T1-C2C',
                    'Avito_Goods_gc-24-02-03-SP-maintenance-SP-T1-C2C',
                    'Avito_Goods_gc-25-01-SP-Garage-2-T1-2C'
                  ],
                  'SP-Tires': [
                    'Avito_Goods_gc-24-10-SP-WinterTires-T1-2C',
                    'Avito_Goods_gc-24-03-04-SP-Summer-tires-T1-C2C'
                  ],
                  'Sale': [
                    'Avito_Goods_gc-24-11-СС-November-BIG-SALE-T1',
                    'Avito_Goods_gc-24-06-СС-SALE-Federal',
                    'Avito_Goods_gc-24-12-EL&LS-SALE-T1'
                      
                  ],
                  'EL': [
                    'Avito_Goods_gc-24-02-EL-GenderHolidays-T1-C2C',
                    'Avito_Goods_gc-24-12-EL&LS-SALE-T1(EL)'
                  ],
                  'LS': [
                    'Avito_Goods_gc-24-04-FS-Avito-Premium-T1-C2C',
                    'Avito_Goods_gc-24-06-СС-SALE-Federal',
                    'Avito_Goods_gc-24-12-EL&LS-SALE-T1'
                  ],
                  'Jobs': [
                    #'Avito_Job_jc-24-08-General-Better_Matching_on_Avito-T1-B2C',
                    'Avito_Job_jc-24-01-General-Find_your_place-T1-B2C',
                    'Avito_Job_jc-24-06-General-Off_season-T1-B2C'
                ],
                  'RRE': [
                      'Avito_RE_re-24-01-ND_SS-RRE_янв-апр-T1-С2С',
                      'Avito_RE_ re-24-09-RRE_сент-ноя-T1-С2С'
                  ],
                  'STR': [
                      'Avito_RE_re-24-10-STR-окт-дек-T1-С2С',
                      'Avito_RE_re-24-04-STR-T1-С2С'
                  ],
                  'LTR': [
                      'Avito_RE_re-24-08-LTR-авг-сент-T1-С2С'
                  ],
                  'Services': [
                      'Avito_Services_se_24-08-CROSS_MR&TR-T1-С2С',
                      'Avito_Services_se_24-06-HH-T1-С2С'
                  ],
                  'Auto': [
                      'Avito_Auto_au-24-09-SL-Select-T1-C2C',
                      'Avito_Auto_au-24-10-NCB-Buyers-T1-C2C'
                  ],
                  'Sellers': [
                      'Avito_Goods_gc-Sellers-25'
                  ]
                }

        self.dataset_creator = DatasetCreator(c_engine, t_engine, for_media=True)
        #self.mvc = MetricValueCalculation(t_engine)

    def get_df(self):
        df_campaigns = self.parser.read_sheet(self.campaigns_path, 'data Nat TV')
        df_metrics = self.parser.read_sheet(self.metrics_path, 'results')

        df_campaigns['SOV\nAll 18-54'] = df_campaigns['SOV\nAll 18-54'].apply(lambda x: float(x[:-1]) / 100)
        df_campaigns['TRPs TA actual'] = df_campaigns['TRPs TA actual'].apply(lambda x: 
                                                                        (int(x.replace(u'\xa0', u'')) // 250) * 250)
        df_campaigns = df_campaigns[['Campaign', 'TRPs TA actual', 'SOV\nAll 18-54', 'budget actual, Net']]
        df_campaigns.columns = ['flight', 'TRP', 'SOV', 'budget actual, Net']

        df = df_campaigns.merge(df_metrics, on='flight', how='inner', suffixes=('', '_drop'))
        df = df[[col for col in df.columns if not col.endswith('_drop')]]
        
        df['base_dtb'] = pd.to_numeric(df['base_dtb'].replace(',', '.', regex=True), errors='coerce')
        df['metric_uplift'] = pd.to_numeric(df['metric_uplift'].replace(',', '.', regex=True), errors='coerce')
        df['metric_mde'] = pd.to_numeric(df['metric_mde'].replace(',', '.', regex=True), errors='coerce')
        df['cross_effect'] = pd.to_numeric(df['cross_effect'].replace(',', '.', regex=True), errors='coerce')
        
        df['budget actual, Net'] = (
            df['budget actual, Net']
            .astype(str)
            .str.replace('\xa0', '', regex=False)  # Remove non-breaking spaces
            .str.replace(' ', '')                  # Remove any other spaces
            .str.replace(',', '.')                 # Replace commas with dots (if any)
            .astype(float)                         # Convert to float
        )

        df['metric_abs_analytics'] = df['base_dtb'] * df['metric_uplift']
        df['mde_abs'] = df['base_dtb'] * (df['metric_mde'] * 0.01)
        
        return df
    


    def get_seen_array(self):
        if self.seen_curves is None:
            with open('/Users/asekorneev/Documents/Work projects/Доки для кода/curves_mean.pickle', 'rb') as handle:
                self.seen_curves = pickle.load(handle)
        return self.seen_curves[self.question]
    

    def load_campaign_data(self, df_campaigns=True, df_metrics=True):
        """
        Объединяет данные TRP/SOV с метриками DTB. 
        """
        
        if df_campaigns:
            self.campaigns_path = input("Введите путь к файлу с кампаниями: ")
            
        if df_metrics:
            self.metrics_path = input("Введите путь к файлу с метриками: ")

        self.get_df()

    def load_reach_trp(self, sheet_id):
        """
        Парсит TRP → охват по колонкам и сохраняет в self.reach_curves.
        """
   
        for flight in self.df['flight']:
            df_raw = self.parser.read_sheet(sheet_id, flight)
            df_curve_raw = df_raw.copy()
            if len(df_curve_raw.columns) > 0:
                df_curve_raw.columns = ['TRP'] + df_curve_raw.columns[1:].tolist()
            else:
                print(f"Ошибка: Пустой DataFrame для {flight}. Проверь данные в Google Sheet.")
                return

            
            df_curve_raw['TRP'] = df_curve_raw['TRP'].replace(',', '.', regex=True)
            df_curve_raw['TRP'] = pd.to_numeric(df_curve_raw['TRP'], errors='coerce')
            df_curve_raw = df_curve_raw.dropna(subset=['TRP'])  
            df_curve_raw['TRP'] = df_curve_raw['TRP'].astype(int)
            
            for col in df_curve_raw.columns[1:]:
                df_curve_raw[col]=df_curve_raw[col].replace(',', '.', regex=True)
                df_curve_raw[col] = df_curve_raw[col].astype(float)
            
            new_cols = ['TRP'] + [int(col[:-1]) for col in df_curve_raw.columns[1:]]  # если колонки были '1%','2%',…
            df_curve_raw.columns = new_cols
            df_curve_raw = df_curve_raw.set_index('TRP', drop=False)
            self.flight_trp_dict[flight] = df_curve_raw

    

    def make_sov_trp_coeffs(self, flight_params, flight, vertical):
        if vertical == 'Goods':
            min_sov = self.min_sov
            max_sov = self.max_sov
        else:
            min_sov = 0.4
            max_sov = 0.56

        trp, sov, metric_abs, mde_abs = flight_params
        competitors_TRP = trp/sov - trp

        self.start_TRP = min_sov * competitors_TRP / (1 - min_sov)
        self.end_TRP   = max_sov * competitors_TRP / (1 - max_sov)

        start_coeff = self.sov_coeff_start
        end_coeff   = self.sov_coeff_end

        sov_trp_coeffs = {}
        sov_values = {}
        for t in sorted(self.flight_trp_dict[flight]['TRP']):
            if t < self.start_TRP:
                curr = start_coeff * t / self.start_TRP
            elif t < self.end_TRP:
                curr = start_coeff + (end_coeff - start_coeff) * (t - self.start_TRP)/(self.end_TRP - self.start_TRP)
            else:
                curr = end_coeff
            sov_trp_coeffs[t] = curr
            sov_val = t / (trp + competitors_TRP)
            
            if sov_val <= 1:
                sov_values[t] = sov_val
            else:
                sov_values[t] = 1.0

        df = self.flight_trp_dict[flight]
        df['coeffs'] = df['TRP'].map(sov_trp_coeffs)
        df['SOV'] = df['TRP'].map(sov_values)

    


    def make_correct_reach_array(self, flight, TRP):
        """
        Возвращает reach_array на основе данных и self.real_seen_coeff
        """
        df = self.flight_trp_dict[flight]
        reach_cols_values = [int(col) for col in df.columns if str(col).isdigit()]

        row = df[df['TRP'] == TRP]
        if row.empty:
            raise ValueError(f"TRP {TRP} не найден для flight {flight}")

        curr_reaches = row[reach_cols_values].values.flatten()

        # Корректируем значения
        corrected_reach_values = np.ceil(np.array(reach_cols_values) * self.real_seen_coeff)

        reaches_df = pd.DataFrame([corrected_reach_values, curr_reaches]).T.groupby(0)[1].max().reset_index()
        corrected_reach_values = np.array(reaches_df[0]).astype(int)
        corrected_reaches = np.array(reaches_df[1])

        reach_array = np.zeros(corrected_reach_values.max() + 1)

        for ind in range(len(corrected_reach_values) - 1):
            currs = (corrected_reaches[ind] - corrected_reaches[ind + 1]) / \
                    (corrected_reach_values[ind + 1] - corrected_reach_values[ind])

            for i in range(corrected_reach_values[ind], corrected_reach_values[ind + 1]):
                reach_array[i] = currs

        reach_array[-1] = corrected_reaches[-1]

        return reach_array


    def make_equal_size_array(self, a, b):
        N = len(a)
        M = len(b)
        
        if M < N:
            b = np.array(list(b) + [b[-1]] * (N - M))
        return a, b


    def predict_metric(self, flight, TRP):
        """
        Возвращает предсказанную метрику для кампании.
        """
        sov_coeff = self.flight_trp_dict[flight]['coeffs']
        reach_array = self.make_correct_reach_array(flight, TRP)
        seen_metric_array = self.get_seen_array()
        
        reach_array, seen_metric_array = self.make_equal_size_array(reach_array, seen_metric_array) 
        metric_val = 0
        for p, val in zip(reach_array, seen_metric_array):
            metric_val += p/100 * val
        return metric_val * sov_coeff[TRP]
    

    def find_fin_coeff(self, start_rk_list):
        coeffs = []
        for flight in start_rk_list:
            try:
                row = self.df[self.df['flight'] == flight].iloc[0]
            except IndexError:
                print(f"Warning: flight {flight} not found in df")
                continue
            #row = self.df[self.df['flight']==flight].iloc[0]
            flight_params = (
                int(row['TRP']),
                float(row['SOV']),
                100.0 if float(row['metric_abs_analytics']) == 0 else float(row['metric_abs_analytics']),
                float(row['mde_abs'])
            )

            self.make_sov_trp_coeffs(flight_params, flight, row['vertical'])
            fin_coeff = self.predict_coeff(flight, flight_params)

            metric_abs = flight_params[2]
            mde_abs    = flight_params[3]
            base       = float(row['base_dtb'])
                
            max_c = fin_coeff * (metric_abs + mde_abs)/metric_abs/base
            min_c = fin_coeff * (metric_abs - mde_abs)/metric_abs/base

            coeffs += list(np.linspace(min_c, max_c, 1000))

        return np.median(coeffs)
    

    def predict_dtb_for_flight(self, flight, start_rk_list, campaigns_type, show_res=False):
        
        if campaigns_type == 'New':
            df = self.data_to_predict
        elif campaigns_type == 'Old':
            df = self.df
        else:
            raise ValueError(f"Campaigns '{campaigns_type}' недопустим")

        row = (df.loc[df['flight'] == flight].iloc[0]) 
        flight_params = (
            int(row['TRP']), float(row['SOV']),
            float(row['metric_abs_analytics']), float(row['mde_abs'])
        )
        self.make_sov_trp_coeffs(flight_params, row['flight'], row['vertical'])
        fin_coeff = self.find_fin_coeff(start_rk_list)

        fact_value   = float(row['metric_abs_analytics'])
        base         = float(row['base_dtb'])
            
        prediction = self.predict_coeff(flight, flight_params, coeff=fin_coeff * base)
        
        return prediction, fact_value
    

    def predict_dtb(self, campaigns_type, show_res=False):

        if campaigns_type == 'New':
            data = self.data_to_predict
        elif campaigns_type == 'Old':
            data = self.df
        else:
            raise ValueError(f"Campaigns '{campaigns_type}' недопустим")
        
        predictions = []
        facts = []
        diff_dict = {}

        for flight in self.campaigns_list:
            start_rk_list = self.get_start_rk_list(flight, data)
            prediction, fact = self.predict_dtb_for_flight(flight, start_rk_list, campaigns_type)
            predictions.append(prediction)
            facts.append(fact)
            diff_dict[flight] = prediction - fact

        if show_res:
            self.visualize_flight_data(self.flight_trp_dict, self.df)
        
        return {
            'plan': np.array(predictions),
            'fact': facts,
            'diff_dict': diff_dict
        }



    def get_start_rk_list(self, flight, data):
        # 1. Достаём скаляр, а не Series
        category = (
            data.loc[data["flight"] == flight, "category"]
            .iat[0]            # или .iloc[0] / .squeeze()
        )

        # 2. Берём две связанные кампании (если вдруг их больше — обрезаем до 2)
        start_rk_list = self.closest_start_rk[category]
        return start_rk_list



    def predict_coeff(self, flight, flight_params, coeff=None):

        trp, sov, metric_abs, mde_abs = flight_params

        trp_x = []
        trp_y = []
        for t in sorted(self.flight_trp_dict[flight]['TRP'].astype(int)):
            trp_x.append(t)
            trp_y.append(self.predict_metric(flight, t))

        if trp_x[-1] < trp:
            trp_x.append(trp)
            trp_y.append(trp_y[-1])
        
        trp_y = np.array(trp_y) / np.max(trp_y)
        trp_x = np.array(trp_x)
     
        if coeff is None:
            start_val = trp_y[trp_x == trp]
            coeff = (metric_abs / start_val) / self.get_creative_coeff(flight)
            trp_y *= coeff
        else:
            trp_y *= coeff * (self.opm * (self.cons/100)) / 6.42
            start_val = trp_y[trp_x == trp]
            coeff = start_val
            
            if 'DTB_pred' in self.flight_trp_dict[flight].columns:
                self.flight_trp_dict[flight].drop(columns=['DTB_pred'], inplace=True)

            # Создаём новый DataFrame с прогнозом
            df_dtb = pd.DataFrame({
                'TRP': trp_x,
                'DTB_pred': trp_y
            })

            # Объединяем
            self.flight_trp_dict[flight] = pd.merge(
                self.flight_trp_dict[flight],
                df_dtb,
                left_index=True,
                right_on='TRP',
                how='left'
            )

            self.flight_trp_dict[flight].drop(columns=['TRP_y', 'TRP_x'], inplace=True)
            self.flight_trp_dict[flight].set_index('TRP', drop=False, inplace=True)

        return coeff
    


    def visualize_flight_data(self, dict_flight, df, *, streamlit_container=None, title=None):
        """
        Рисует предсказание по кампании(ям) + фактические точки и 95% CI (колонки low/high).
        Светло-голубая лента CI строится как полигон между low и high.
        Если TRP одновременно и индекс, и колонка — приводим к формату: TRP = колонка.
        """

        # Какие кампании рисуем
        flight_names = list(self.campaigns_list) or list(dict_flight.keys())

        # Палитра
        colors = px.colors.qualitative.Plotly
        color_map = {f: colors[i % len(colors)] for i, f in enumerate(flight_names)}

        fig = go.Figure()

        for flight_name in flight_names:
            raw = dict_flight.get(flight_name)
            if raw is None or len(raw) == 0:
                continue

            # ---- Нормализуем df_flight: TRP всегда колонка, индекс без имени ----
            df_flight = raw.copy()
            if df_flight.index.name == "TRP":
                df_flight = df_flight.reset_index()
            if "TRP" not in df_flight.columns:
                df_flight = df_flight.rename_axis("TRP").reset_index()
            # убираем возможные дубликаты колонок
            df_flight = df_flight.loc[:, ~df_flight.columns.duplicated()]

            # приведение типов и сортировка
            tmp = df_flight.copy()
            for col in ["TRP", "DTB_pred", "low", "high", "SOV"]:
                if col in tmp.columns:
                    tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
            tmp = tmp.dropna(subset=["TRP"]).sort_values("TRP")

            color = color_map.get(flight_name, "#1f77b4")

            # ---- ЛЕНТА CI (если есть low/high) ----
            if {"low", "high"}.issubset(tmp.columns):
                band = tmp[["TRP", "low", "high"]].dropna()
                if not band.empty:
                    # гарантируем low <= high для каждого TRP
                    mask = band["low"] > band["high"]
                    if mask.any():
                        band.loc[mask, ["low", "high"]] = band.loc[mask, ["high", "low"]].to_numpy()

                    x_poly = np.concatenate([band["TRP"].to_numpy(),
                                            band["TRP"].to_numpy()[::-1]])
                    y_poly = np.concatenate([band["high"].to_numpy(),
                                            band["low"].to_numpy()[::-1]])

                    fig.add_trace(go.Scatter(
                        x=x_poly, y=y_poly,
                        mode="lines",
                        line=dict(width=0),
                        fill="toself",
                        fillcolor="rgba(0,153,255,0.20)",  # светло-голубая полупрозрачная
                        hoverinfo="skip",
                        name=f"{flight_name} • 80% CI",
                        showlegend=True
                    ))

            # ---- ЛИНИЯ ПРЕДСКАЗАНИЯ ----
            if "DTB_pred" in tmp.columns and tmp["DTB_pred"].notna().any():
                fig.add_trace(go.Scatter(
                    x=tmp["TRP"], y=tmp["DTB_pred"],
                    mode="lines",
                    name=f"{flight_name} • Predicted",
                    line=dict(color=color, width=2),
                ))

                # Точки предсказания + hover по SOV (если есть)
                if "SOV" in tmp.columns:
                    hov = None
                    if tmp["SOV"].notna().any():
                        max_sov = tmp["SOV"].max()
                        hov = [f"SOV: {v*100:.2f}%" if max_sov <= 1 else f"SOV: {v:.2f}%"
                            for v in tmp["SOV"].fillna(np.nan)]
                    fig.add_trace(go.Scatter(
                        x=tmp["TRP"], y=tmp["DTB_pred"],
                        mode="markers",
                        name=f"{flight_name} • Predicted pts",
                        marker=dict(color=color, size=6, symbol="circle"),
                        text=hov,
                        hovertemplate=('%{text}<br>TRP: %{x}<br>DTB_pred: %{y}<extra></extra>'
                                    if hov else None),
                        showlegend=False
                    ))

            # ---- ФАКТИЧЕСКИЕ ТОЧКИ (с ошибками, если есть) ----
            if isinstance(df, pd.DataFrame) and "flight" in df.columns:
                df_actual = df[df["flight"] == flight_name]
                if not df_actual.empty and "metric_abs_analytics" in df_actual.columns:
                    y_act = pd.to_numeric(df_actual["metric_abs_analytics"], errors="coerce")
                    if y_act.notna().any():
                        err = None
                        if "mde_abs" in df_actual.columns:
                            err = pd.to_numeric(df_actual["mde_abs"], errors="coerce")
                        fig.add_trace(go.Scatter(
                            x=pd.to_numeric(df_actual["TRP"], errors="coerce"),
                            y=y_act,
                            mode="markers",
                            name=f"{flight_name} • Actual",
                            marker=dict(color=color, size=8, symbol="diamond"),
                            error_y=dict(
                                type="data",
                                array=err if err is not None else None,
                                visible=bool(err is not None and err.notna().any())
                            )
                        ))

        title_txt = title or (f"{flight_names[0]} — DTB" if flight_names else "DTB")
        fig.update_layout(
            title=dict(text=title_txt, x=0.01, xanchor="left"),
            xaxis_title="TRP",
            yaxis_title="DTB",
            template="plotly_white",
            height=650,
            legend=dict(x=0.01, y=0.98, bgcolor="rgba(0,0,0,0)")
        )

        # вывод в Streamlit, если передан контейнер (обычно это st)
        if streamlit_container is None:
            try:
                import streamlit as st
                streamlit_container = st
            except Exception:
                streamlit_container = None

        if streamlit_container is not None:
            streamlit_container.plotly_chart(fig, use_container_width=True)
        else:
            fig.show()

        return fig




    def trp_to_budget(self, flight):

        row = self.df[self.df['flight'] == flight]
        if row.empty:
            raise ValueError(f"Flight '{flight}' not found in df")

        curr_budget = row['budget actual, Net'].values[0]
        curr_trp = row['TRP'].values[0]

        self.flight_trp_dict[flight]['budget'] = self.flight_trp_dict[flight]['TRP'].astype(float).apply(
            lambda trp: self.ttb_coeff(curr_trp, trp, curr_budget)
    )


        
    def dtb_to_ROMI(self, flight):

        long_term_effect = 2
        cross_effect = float(self.df[self.df['flight'] == flight]['cross_effect'])
        dtb_value = float(self.df[self.df['flight'] == flight]['metric_value'])

        df = self.flight_trp_dict[flight]
        self.flight_trp_dict[flight]['ROMI'] = (
            (df['DTB_pred'] * dtb_value * cross_effect * long_term_effect) / df['budget'] - 1
        )

    
    def predict_ROMI(self, campaigns_type, show_res=False):

        def safe_float(value):
            try:
                return float(value)
            except (ValueError, TypeError):
                return np.nan

        self.predict_dtb(campaigns_type, show_res=False)

        romis = {}
        romi_pred = []
        romi_fact = []
        romi_best = []

        if campaigns_type == 'New':
            data = self.data_to_predict
        elif campaigns_type == 'Old':
            data = self.df
        else:
            raise ValueError(f"Campaigns '{campaigns_type}' недопустим")

        for flight in self.campaigns_list:
            self.trp_to_budget(flight)
            self.dtb_to_ROMI(flight)

            if campaigns_type == 'Old':
                fact_ROMI = data[data['flight'] == flight]['romi']
            else:
                fact_ROMI = 0

            df_flight = self.flight_trp_dict[flight]
            trp_value = data[data['flight'] == flight]['TRP'].iloc[0]
            pred_ROMI = df_flight[df_flight['TRP'] == trp_value]['ROMI']

            best_ROMI = -1
            best_TRP = 0
            for _, row in df_flight.iterrows():
                if row['ROMI'] > best_ROMI:
                    best_ROMI = row['ROMI']
                    best_TRP = row['TRP']

            romis[flight] = [best_ROMI, best_TRP]
            romi_best.append(best_ROMI)
            romi_fact.append(safe_float(fact_ROMI.values[0]) if len(fact_ROMI) > 0 else np.nan)
            romi_pred.append(safe_float(pred_ROMI.values[0]) if len(pred_ROMI) > 0 else np.nan)

        if show_res:
            self.visualize_flight_data(self.flight_trp_dict, data)

        return {
            'tuple': romis, 
            'prediction': np.array(romi_pred, dtype=float), 
            'fact': np.array(romi_fact, dtype=float), 
            'best': np.array(romi_best, dtype=float)
        }

    def get_creative_coeff(self, flight, avr_coeff=None):
        self.df['OPM'] = pd.to_numeric(self.df['OPM'].replace(',', '.', regex=True), errors='coerce')
        opm = self.df.loc[self.df['flight'] == flight, 'OPM'].values[0]
        
        self.df['consideration'] = pd.to_numeric(self.df['consideration'].replace(',', '.', regex=True), errors='coerce')
        cons = self.df.loc[self.df['flight'] == flight, 'consideration'].values[0]
        
        vertical = self.df.loc[self.df['flight'] == flight, 'vertical'].values[0]
        if vertical == 'Goods':
            avr_coeff = 6.42
        elif vertical == 'Auto':
            avr_coeff = 6.89
        elif vertical == 'Services':
            avr_coeff = 8
        elif vertical == 'Jobs':
            avr_coeff = 5.22
        else:
            avr_coeff = 8
        
        cc = opm * cons
        if pd.isna(opm) or pd.isna(cons):
            creative_coeff = 1
        else:
            creative_coeff = cc / avr_coeff

        return creative_coeff
    
    
    def get_logical_categories(self, df, flight) -> list[str]:
        
        series = df.loc[df["flight"] == flight, "logical_category"].dropna()

        cats = {
            cat.strip()                      # убираем пробелы вокруг
            for cell in series
            for cat in str(cell).split(",")  # строка → отдельные категории
            if cat.strip()                   # отбрасываем пустые
        }

        return cats                  # список, если нужен порядок

    
    def get_params_for_flight(self, flight):

        row = self.data_to_predict[self.data_to_predict['flight'] == flight].iloc[0]
        date_start = pd.to_datetime(row['date_start'])
        date_end   = pd.to_datetime(row['date_end'])
        category = row['category']

        main_params = {
            'flight_name': flight,
            'flight_start_date': date_start.strftime('%Y-%m-%d'),
            'flight_end_date': date_end.strftime('%Y-%m-%d'),
            'analysed_start_date': date_start.strftime('%Y-%m-%d'),
            'analysed_end_date': date_end.strftime('%Y-%m-%d'),
            'flight_budget': 0,
            'metrics': ['DTB'],
            'test regions': ['Any'],
            'control regions': [],
            'exclude regions': [],
        }

        slices = {
            f'{category}': {
                'logical_category': self.get_logical_categories(self.data_to_predict, flight),
                'vertical': list(np.unique(self.data_to_predict[self.data_to_predict['flight'] == flight]['vertical']))
            }
        }

        return main_params, slices
    

    def get_dtb_bases(self, data):

        for flight in data['flight'].values:
            mask = self.data_to_predict['flight'] == flight
            if not mask.any():
                raise ValueError(f"Flight '{flight}' отсутствует в data_to_predict")

            row = self.data_to_predict.loc[mask].iloc[0]
            
            date_start = pd.to_datetime(row['date_start'])
            date_end   = pd.to_datetime(row['date_end'])

            main_params, slices = self.get_params_for_flight(flight)
            dc_result = self.dataset_creator.get_datasets(main_params, slices)

            for cat in slices:
                for metric in main_params['metrics']:
                    curr_df = dc_result['ml_datasets'][(metric, cat)]['dataset']
                    dtb_sum = curr_df[
                        (curr_df['date'] >= (date_start - pd.DateOffset(years=2)).strftime('%Y-%m-%d')) &
                        (curr_df['date'] <= (date_end   - pd.DateOffset(years=2)).strftime('%Y-%m-%d'))
                    ]['Any'].sum()

            self.data_to_predict.loc[mask, 'base_dtb'] = dtb_sum
        
    def get_dtb_base_for_flight(self, vertical, logcats, date_start, date_end, flight='Test-flight'):
        # Приводим всё к Timestamp (без времени) для единообразия
        ds = pd.Timestamp(date_start)   # дата начала
        de = pd.Timestamp(date_end)     # дата окончания
        today = pd.Timestamp("today").normalize()

        # Параметры для dataset_creator можно оставить строками
        main_params = {
            'flight_name': {flight},
            'flight_start_date': ds.strftime('%Y-%m-%d'),
            'flight_end_date': de.strftime('%Y-%m-%d'),
            'analysed_start_date': ds.strftime('%Y-%m-%d'),
            'analysed_end_date': de.strftime('%Y-%m-%d'),
            'flight_budget': 0,
            'metrics': ['DTB'],
            'test regions': ['Any'],
            'control regions': [],
            'exclude regions': [],
        }

        slices = {
            'ALL': {
                'logical_category': logcats,      # список строк, например ['Goods.Fashion']
                'vertical': [str(vertical)],      # строго строка 'Goods' внутри списка
            }
        }

        dc_result = self.dataset_creator.get_datasets(main_params, slices)

        yoy_dtb_coeff = 1.0
        metric = 'DTB'
        cat = 'ALL'

        curr_df = dc_result['ml_datasets'][(metric, cat)]['dataset'].copy()

        # Убедимся, что колонка 'date' — datetime64
        if not np.issubdtype(curr_df['date'].dtype, np.datetime64):
            curr_df['date'] = pd.to_datetime(curr_df['date'])

        # --- выбор интервала и расчёты, БЕЗ strftime в масках ---
        if de < today:
            mask = (curr_df['date'] >= ds) & (curr_df['date'] <= de)
            dtb_sum = curr_df.loc[mask, 'Any'].sum()

        elif de - pd.DateOffset(years=1) < today:
            ds1, de1 = ds - pd.DateOffset(years=1), de - pd.DateOffset(years=1)
            mask1 = (curr_df['date'] >= ds1) & (curr_df['date'] <= de1)
            dtb_sum = curr_df.loc[mask1, 'Any'].sum()

            ds2, de2 = ds - pd.DateOffset(years=2), de - pd.DateOffset(years=2)
            mask2 = (curr_df['date'] >= ds2) & (curr_df['date'] <= de2)

            denom = curr_df.loc[mask2, 'Any'].sum()
            num   = curr_df.loc[mask1, 'Any'].sum()
            yoy_dtb_coeff = (num / denom) if denom not in (0, 0.0, np.nan) else 1.0

        else:
            ds2, de2 = ds - pd.DateOffset(years=2), de - pd.DateOffset(years=2)
            mask2 = (curr_df['date'] >= ds2) & (curr_df['date'] <= de2)
            dtb_sum = curr_df.loc[mask2, 'Any'].sum()

            # пример твоей логики YoY с фиксированными годами — тоже только Timestamps
            m1_start = pd.Timestamp(2024, 1, 1)
            m1_end   = today - pd.DateOffset(years=1)
            m2_start = pd.Timestamp(2025, 1, 1)
            m2_end   = today

            mask_m1 = (curr_df['date'] >= m1_start) & (curr_df['date'] <= m1_end)
            mask_m2 = (curr_df['date'] >= m2_start) & (curr_df['date'] <= m2_end)

            denom = curr_df.loc[mask_m2, 'Any'].sum()
            num   = curr_df.loc[mask_m1, 'Any'].sum()
            yoy = (num / denom) if denom not in (0, 0.0, np.nan) else 1.0
            yoy_dtb_coeff = yoy ** 2

        return float(dtb_sum) * float(yoy_dtb_coeff)
    
    # def get_dtb_value_for_flight(self, vertical, logcats, date_start, date_end, base_dtb, flight='Test-flight'):
    #     ds = pd.Timestamp(date_start)   # дата начала
    #     de = pd.Timestamp(date_end)     # дата окончания
    #     today = pd.Timestamp("today").normalize()

    #     # Параметры для dataset_creator можно оставить строками
    #     main_params = {
    #         'flight_name': {flight},
    #         'flight_start_date': ds.strftime('%Y-%m-%d'),
    #         'flight_end_date': de.strftime('%Y-%m-%d'),
    #         'analysed_start_date': ds.strftime('%Y-%m-%d'),
    #         'analysed_end_date': de.strftime('%Y-%m-%d'),
    #         'flight_budget': 0,
    #         'metrics': ['DTB'],
    #         'test regions': ['Any'],
    #         'control regions': [],
    #         'exclude regions': [],
    #     }
    #     slices = {
    #         'ALL': {
    #             'logical_category': logcats,  
    #             'vertical': [str(vertical)],
    #         }
    #     }

    #     yoy_dtb_coeff = 1.0
    #     metric = 'DTB'
    #     cat = 'ALL'
    #     elasticity_data = self.mvc.get_liquidity_metric_values(main_params, slices)

    #     curr = elasticity_data[(cat, metric)]
    #     curr_value = int(curr['total_elasticity'] * curr['total_revenue'] / base_dtb)
        
        
    def get_dtb_values(self, flight):
        row = self.data_to_predict[self.data_to_predict['flight'] == flight].iloc[0]
        date_start = row['date_start'].strftime('%Y-%m-%d')
        date_end = row['date_end'].strftime('%Y-%m-%d')

        main_params = {
            'flight_name': flight,
            'flight_start_date': date_start,
            'flight_end_date': date_end,
            'analysed_start_date': date_start,
            'analysed_end_date': date_end,
            'flight_budget': 0,
            'metrics': ['DTB', 'buyers'],
            'test regions': ['Any'],
            'control regions': [],
            'exclude regions': [],
        }

        slices = {
            'ALL': {
                'logical_category': list(np.unique(self.data_to_predict[self.data_to_predict['flight'] == flight]['logical_category'])),
                'vertical': list(np.unique(self.data_to_predict[self.data_to_predict['flight'] == flight]['vertical']))
            }
        }

        dc = self.dataset_creator
        dc_result = dc.get_datasets(main_params, slices)

        mvc = self.metric_value_calculation
        elasticity_data = mvc.get_liquidity_metric_values(main_params, slices)

        metric_vals = {}
        bases = {}

        for cat in slices:
            for metric in main_params['metrics']:
                curr_df = dc_result['ml_datasets'][(metric, cat)]['dataset']
                dtb_sum = curr_df[
                    (curr_df['date'] >= date_start) &
                    (curr_df['date'] <= date_end)
                ]['Any'].sum()

                curr = elasticity_data[(cat, 'buyers')]
                curr_value = int(curr['total_elasticity'] * curr['total_revenue'] / dtb_sum)
                metric_vals[(cat, metric)] = curr_value
                bases[(cat, metric)] = dtb_sum

        # Обновляем значения в self.data_to_predict
        self.data_to_predict.loc[self.data_to_predict['flight'] == flight, 'dtb_value'] = metric_vals[('ALL', 'DTB')]
        self.data_to_predict.loc[self.data_to_predict['flight'] == flight, 'buyer_value'] = metric_vals[('ALL', 'buyers')]
        self.data_to_predict.loc[self.data_to_predict['flight'] == flight, 'dtb_base'] = bases[('ALL', 'DTB')]

        return metric_vals[('ALL', 'DTB')], metric_vals[('ALL', 'buyers')], bases[('ALL', 'DTB')]

    
    def get_cross_effect(self, flight):
        data_dict = self.cross_effect.get_data_for_crosseffect() 

        cross_eff = self.cross_effect.get_crosseffect_with_calculated_data(data_dict, list(np.unique(self.data_to_predict[self.data_to_predict['flight'] == flight]['logical_category'])))
        self.data_to_predict.loc[self.data_to_predict['flight'] == flight, 'cross_effect'] = cross_eff

        return cross_eff
    

    def add_to_trp_dict(self, data):

        # гарантируем тип datetime (на случай строковых столбцов)
        self.df["date_start"] = pd.to_datetime(self.df["date_start"])
        data = data.copy()
        data["date_start"] = pd.to_datetime(data["date_start"])

        for _, new_row in data.iterrows():
            new_flight = new_row["flight"]
            cat = new_row["category"]
            new_ds = new_row["date_start"]

            # 1. выбираем старые кампании той же категории
            old_cat = self.df[self.df["category"] == cat]

            if old_cat.empty:
                raise ValueError(
                    f"В категории '{cat}' нет старых кампаний для наследования "
                    f"кривой (flight '{new_flight}')."
                )

            # 2. находим ближайшую по дате start
            idx = (old_cat["date_start"] - new_ds).abs().idxmin()
            nearest_flight = old_cat.loc[idx, "flight"]

            # 3. копируем кривую TRP→Reach
            if nearest_flight not in self.flight_trp_dict:
                raise KeyError(
                    f"У старой кампании '{nearest_flight}' нет записи "
                    f"в self.flight_trp_dict."
                )

            # сохраняем в тот же словарь (или self.trp_reach_dict)
            self.flight_trp_dict[new_flight] = (
                self.flight_trp_dict[nearest_flight].copy()
            )
 

        
    def predict_for_discrete_optimizer(self, data, dtb_dict, SOV_dict):
        dict_predictions = {}
        for i in range(1, 13):
            for flight in list(np.unique(data['flight'])):
                row = (data.loc[data['flight'] == flight].iloc[0]) 
                flight_params = (
                    1000, float(SOV_dict[row['logical_category']][i]), 0, 0
                )
                self.make_sov_trp_coeffs(flight_params, row['flight'], row['vertical'])
                start_rk_list = self.get_start_rk_list(flight, data)
                fin_coeff = self.find_fin_coeff(start_rk_list)

                base = int(dtb_dict[row['logical_category']][i])
                    
                prediction = self.predict_coeff_optimize(flight, flight_params, i, coeff=fin_coeff * base)
                dict_predictions[flight + '_' + str(i)] = prediction

        return dict_predictions


        


    def predict_coeff_optimize(self, flight, flight_params, month, coeff=None):

        trp, sov, metric_abs, mde_abs = flight_params

        trp_x = []
        trp_y = []
        for t in sorted(self.flight_trp_dict[flight]['TRP'].astype(int)):
            trp_x.append(t)
            trp_y.append(self.predict_metric(flight, t))

        if trp_x[-1] < trp:
            trp_x.append(trp)
            trp_y.append(trp_y[-1])
        
        trp_y = np.array(trp_y) / np.max(trp_y)
        trp_x = np.array(trp_x)
     
        if coeff is None:
            start_val = trp_y[trp_x == trp]
            coeff = (metric_abs / start_val) / self.get_creative_coeff(flight)
            trp_y *= coeff
        else:
            trp_y *= coeff 
            start_val = trp_y[trp_x == trp]
            coeff = start_val
            
            if 'DTB_pred' in self.flight_trp_dict[flight].columns:
                self.flight_trp_dict[flight].drop(columns=['DTB_pred'], inplace=True)

            # Создаём новый DataFrame с прогнозом
            df_dtb = pd.DataFrame({
                'TRP': trp_x,
                'DTB_pred': trp_y
            })

            # Объединяем
            self.flight_trp_dict[flight + '_' + str(month)] = pd.merge(
                self.flight_trp_dict[flight],
                df_dtb,
                left_index=True,
                right_on='TRP',
                how='left'
            )

            self.flight_trp_dict[flight + '_' + str(month)].drop(columns=['TRP_y', 'TRP_x'], inplace=True)
            self.flight_trp_dict[flight + '_' + str(month)].set_index('TRP', drop=False, inplace=True)

        return coeff


    def predict_ROMI_for_discrete_optimizer(self, data, dtb_dict, SOV_dict, value_dict, trp_cost_dict, CE_dict):

        flights = self.predict_for_discrete_optimizer(data, dtb_dict, SOV_dict).keys()

        for flight in flights:
            self.dtb_to_ROMI_for_discrete_optimizer(data, flight, value_dict, trp_cost_dict, CE_dict)




    def dtb_to_ROMI_for_discrete_optimizer(self, data, flight, value_dict, trp_cost_dict, CE_dict):
        long_term_effect = 2
        
        m = re.match(r'^(.*)_(\d{1,2})$', flight)
        if m:
            base  = m.group(1)
            month = int(m.group(2))
        else:
            base, month = flight, None

        logcat = data.loc[data['flight'] == base, 'logical_category'].values[0]
        category = data.loc[data['flight'] == base, 'category'].values[0]
        
        cross_effect = float(CE_dict[logcat])
        dtb_value = float(value_dict[logcat][month])

        df = self.flight_trp_dict[flight]

        self.flight_trp_dict[flight]['budget'] = df['TRP'] * trp_cost_dict[logcat][month] * 1000
        
        self.flight_trp_dict[flight]['revenue'] = df['DTB_pred'] * dtb_value * cross_effect * long_term_effect

        self.flight_trp_dict[flight]['ROMI'] = (
            df['revenue'] / df['budget'] - 1
        )
        
        logcat_set = set(item.strip() for item in logcat.split(","))
        self.flight_trp_dict[flight]['logcats'] = [logcat_set] * len(df)
        
        self.flight_trp_dict[flight]['category'] = [category] * len(df)
        
    
    def predict_for_discrete_optimizer_bootstrap(self, data, dtb_dict, SOV_dict, n_bootstrap=1000):

        # 1) отбираем только те флайты, которые уже имеют суффикс _<month>
        filtered = {
            fm: df.copy()
            for fm, df in self.flight_trp_dict.items()
            if fm.rsplit('_', 1)[-1].isdigit()
        }

        preds = {
            flight_month: {int(trp): [] for trp in df.index.astype(int)}
            for flight_month, df in filtered.items()
        }

        for i in tqdm(range(n_bootstrap), desc="Bootstrap"):
            for flight_month, profile_df in filtered.items():
                base_flight, month_str = flight_month.rsplit('_', 1)
                month = int(month_str)

                # найдём row по чистому имени кампании
                try:
                    row = data.loc[data['flight'] == base_flight].iloc[0]
                except IndexError:
                    raise KeyError(f"Flight '{base_flight}' not in data")

                cat  = row['logical_category']
                sov  = float(SOV_dict[cat][month])
                base = int(dtb_dict[cat][month])

                # рассчитаем финальный coeff
                flight_params = (1000, sov, 0, 0)
                self.make_sov_trp_coeffs(flight_params, flight_month, row['vertical'])
                start_rk  = self.get_start_rk_list(base_flight, data)
                fin_coeff = self.find_fin_coeff_bootstrap(start_rk)

                # вызываем predict_coeff_optimize — он создаёт в self.flight_trp_dict
                # новый ключ flight_month + '_' + str(i) с DataFrame прогнозов
                _ = self.predict_coeff_optimize(
                    flight_month,
                    flight_params,
                    month=i,
                    coeff=fin_coeff * base
                )

                # теперь та DataFrame лежит под этим ключом
                df_pred = self.flight_trp_dict[f"{flight_month}_{i}"]

                for trp_level in df_pred.index.astype(int):
                    preds[flight_month][trp_level].append(df_pred.at[trp_level, 'DTB_pred'])

        result = {}
        for flight_month, trp_dict in preds.items():
            trps  = sorted(trp_dict.keys())
            lows  = [np.percentile(trp_dict[t],  2.5)  for t in trps]
            highs = [np.percentile(trp_dict[t], 97.5)  for t in trps]

            df_ci = pd.DataFrame({'low': lows, 'high': highs}, index=trps)
            df_ci.index.name = 'TRP'

            full_df = filtered[flight_month].join(df_ci, how='left')
            result[flight_month] = full_df

        return result


    
    def find_fin_coeff_bootstrap(self, start_rk_list):
        coeffs = []
        for flight in start_rk_list:
            try:
                row = self.df[self.df['flight'] == flight].iloc[0]
            except IndexError:
                print(f"Warning: flight {flight} not found in df")
                continue
            raw_res = 10.0 if float(row['metric_abs_analytics']) == 0 else float(row['metric_abs_analytics'])
            mean = raw_res
            sigma = float(row['mde_abs']) / 1.96

            random_value = np.random.normal(loc=mean, scale=sigma)
            
            flight_params = (
                int(row['TRP']),
                float(row['SOV']),
                10.0 if random_value == 0 else random_value,
                float(row['mde_abs'])
            )

            self.make_sov_trp_coeffs(flight_params, flight, row['vertical'])
            fin_coeff = self.predict_coeff(flight, flight_params)
            
            metric_abs = flight_params[2]
            mde_abs    = flight_params[3]
            base       = float(row['base_dtb'])
                
            max_c = fin_coeff * (metric_abs + mde_abs)/metric_abs/base
            min_c = fin_coeff * (metric_abs - mde_abs)/metric_abs/base

            coeffs += list(np.linspace(min_c, max_c, 1000))

        return np.median(coeffs)
    
    def confidence_intervals_for_prediction(self, data, n_bootstrap=500, ci=(10, 90), overwrite_flight_dict=True):
        """
        Bootstrap CI для одной/нескольких кампаний.
        Ожидаемые поля в data: ['flight','vertical','logical_category','SOV','base_dtb','TRP'].

        Возвращает:
        - DataFrame ['TRP','DTB_pred','low','median','high','SOV'] для одной кампании,
        - либо dict {flight: DataFrame} для нескольких.
        Центральная кривая берётся как медиана bootstrap.
        """
        import numpy as np
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame([data])

        flights = list(getattr(self, "campaigns_list", []) or pd.unique(data["flight"]))
        result = {}

        for flight in flights:
            try:
                row = data.loc[data["flight"] == flight].iloc[0]
            except IndexError:
                raise KeyError(f"Flight '{flight}' not found in data")

            vertical = str(row["vertical"])
            sov_in   = float(row["SOV"])
            base     = float(row["base_dtb"])
            trp_in   = int(row["TRP"])

            preds = None
            trp_grid = None

            for i in range(n_bootstrap):
                flight_params = (trp_in, sov_in, 0, 0)
                self.make_sov_trp_coeffs(flight_params, flight, vertical)
                start_rk  = self.get_start_rk_list(flight, data)
                fin_coeff = self.find_fin_coeff_bootstrap(start_rk)

                _ = self.predict_coeff(flight, flight_params, coeff=fin_coeff * base)
                df_pred = self.flight_trp_dict[flight].copy()

                if trp_grid is None:
                    idx = pd.to_numeric(df_pred.index, errors="coerce")
                    trp_grid = sorted(pd.unique(idx.dropna().astype(int)).tolist())
                    preds = np.full((n_bootstrap, len(trp_grid)), np.nan, dtype=float)

                cur_idx = pd.to_numeric(df_pred.index, errors="coerce").astype("Int64")
                cur_y   = pd.to_numeric(df_pred.get("DTB_pred"), errors="coerce")

                aligned = pd.Series(np.nan, index=trp_grid, dtype=float)
                mask = cur_idx.notna() & cur_y.notna()
                cur_keys = [int(v) for v in cur_idx[mask].tolist()]
                tmp = pd.Series(cur_y[mask].to_numpy(), index=cur_keys, dtype=float).groupby(level=0).mean()
                common = aligned.index.intersection(tmp.index)
                if len(common) > 0:
                    aligned.loc[common] = tmp.loc[common].values

                preds[i, :] = aligned.values

            # --- агрегируем интервалы ---
            low    = np.nanpercentile(preds, ci[0], axis=0)
            median = np.nanpercentile(preds, 50.0,  axis=0)
            high   = np.nanpercentile(preds, ci[1], axis=0)

            # --- выравниваем SOV по той же сетке TRP (берём из последнего df в словаре) ---
            sov_aligned = None
            df_last = self.flight_trp_dict.get(flight)
            if df_last is not None and "SOV" in df_last.columns:
                idx_last = pd.to_numeric(df_last.index, errors="coerce").astype("Int64")
                sov_vals = pd.to_numeric(df_last["SOV"], errors="coerce")

                sov_aligned = pd.Series(np.nan, index=trp_grid, dtype=float)
                mask = idx_last.notna() & sov_vals.notna()
                keys = [int(v) for v in idx_last[mask].tolist()]
                tmp_sov = pd.Series(sov_vals[mask].to_numpy(), index=keys, dtype=float).groupby(level=0).mean()
                common = sov_aligned.index.intersection(tmp_sov.index)
                if len(common) > 0:
                    sov_aligned.loc[common] = tmp_sov.loc[common].values

            # если по какой-то причине SOV не удалось выровнять — используем исходный sov_in как fallback
            if sov_aligned is None:
                sov_aligned = pd.Series([sov_in] * len(trp_grid), index=trp_grid, dtype=float)

            # --- итоговая таблица ---
            out = pd.DataFrame({
                "TRP": trp_grid,
                "low": low,
                "median": median,
                "high": high,
                "SOV": sov_aligned.to_numpy()
            })

            result[flight] = out
            if overwrite_flight_dict:
                self.flight_trp_dict[flight] = out.set_index("TRP")

        return result[flights[0]] if len(flights) == 1 else result






        
