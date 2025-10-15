import streamlit as st
from datetime import date
from itertools import chain
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pickle

sidebar_nav(active_team="media")

# ----- пути к библиотекам -----
lib_dir = '/Users/asekorneev/Documents/Work projects/Код'
if lib_dir not in sys.path:
    sys.path.append(lib_dir)
from pROMIser_for_one_flight import pROMIser_t
from password import password

lib_dir = '/Users/asekorneev/Downloads/Lib'
if lib_dir not in sys.path:
    sys.path.append(lib_dir)
from Google.googlya import Googlya
from Google.TableParser import GoogleSheetsParser
import bxtools.clickhouse as clickhouse
import bxtools.trino as trino

# ----- DWH connectors -----
def connect_dwh(password):
    c_engine = clickhouse.CHEngine(user='asekorneev', clickhouse_password=password)
    t_engine = trino.TrinoEngine(user='asekorneev', password=password)
    return c_engine, t_engine

def visualize_ci_prediction(ci, actual_df=None, *, streamlit_container=None,
                            title=None, flight_name=None):
    """
    Рисует предсказание(я) с 95% CI-лентой и фактическими точками.

    Parameters
    ----------
    ci : pandas.DataFrame | dict[str, pandas.DataFrame]
        Таблица(ы) с колонками минимум: 'TRP' и одна из ['DTB_pred', 'median'].
        Для ленты CI нужны ещё 'low' и 'high'. Доп. колонка 'SOV' — для hover.
    actual_df : pandas.DataFrame | None
        Исторические данные; ожидает колонки 'flight', 'TRP',
        'metric_abs_analytics', опционально 'mde_abs'.
    streamlit_container : module-like | None
        Если передать streamlit (обычно `st`), покажет график через st.plotly_chart.
        Иначе вызовет fig.show().
    title : str | None
        Заголовок графика.
    flight_name : str | None
        Имя кампании для случая, когда `ci` — одиночный DataFrame и
        нужно отфильтровать факт по конкретному flight из `actual_df`.
    """

    def _normalize_df(df):
        """TRP → колонка, всё числовое где нужно, сортируем по TRP."""
        out = df.copy()
        if "TRP" not in out.columns:
            if out.index.name == "TRP":
                out = out.reset_index()
            else:
                out = out.rename_axis("TRP").reset_index()
        out = out.loc[:, ~out.columns.duplicated()]
        for c in ["TRP", "DTB_pred", "median", "low", "high", "SOV"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.dropna(subset=["TRP"]).sort_values("TRP")
        return out

    # Подготовим вход — единообразно итерироваться по (name, df_ci)
    if isinstance(ci, dict):
        items = list(ci.items())
    else:
        name = flight_name
        if name is None:
            name = title.split(" — ", 1)[0] if isinstance(title, str) and " — " in title else "Flight"
        items = [(name, ci)]

    colors = px.colors.qualitative.Plotly
    fig = go.Figure()

    for i, (name, df_ci) in enumerate(items):
        df_ci = _normalize_df(df_ci)
        if df_ci.empty:
            continue

        color = colors[i % len(colors)]

        # --- ЛЕНТА CI (low/high) ---
        if {"low", "high"}.issubset(df_ci.columns):
            band = df_ci[["TRP", "low", "high"]].dropna()
            if len(band):
                # гарантируем low <= high на каждой точке
                swap_mask = band["low"] > band["high"]
                if bool(swap_mask.any()):
                    band.loc[swap_mask, ["low", "high"]] = band.loc[swap_mask, ["high", "low"]].to_numpy()

                x_poly = np.concatenate([band["TRP"].to_numpy(),
                                         band["TRP"].to_numpy()[::-1]])
                y_poly = np.concatenate([band["high"].to_numpy(),
                                         band["low"].to_numpy()[::-1]])

                fig.add_trace(go.Scatter(
                    x=x_poly, y=y_poly,
                    mode="lines",
                    line=dict(width=0),
                    fill="toself",
                    fillcolor="rgba(0,153,255,0.20)",  # светло-голубая лента
                    hoverinfo="skip",
                    name=f"{name} • 80% CI",
                    showlegend=True,
                ))

        # --- ЛИНИЯ ПРЕДСКАЗАНИЯ ---
        ycol = "DTB_pred" if "DTB_pred" in df_ci.columns and df_ci["DTB_pred"].notna().any() else \
               "median"   if "median"   in df_ci.columns and df_ci["median"].notna().any()   else None
        if ycol is not None:
            fig.add_trace(go.Scatter(
                x=df_ci["TRP"],
                y=df_ci[ycol],
                mode="lines",
                name=f"{name} • Predicted",
                line=dict(color=color, width=2),
            ))

            # точки + hover SOV (если есть)
            hov = None
            if "SOV" in df_ci.columns and df_ci["SOV"].notna().any():
                max_sov = df_ci["SOV"].max()
                if max_sov <= 1:
                    hov = [f"SOV: {v*100:.2f}%" for v in df_ci["SOV"]]
                else:
                    hov = [f"SOV: {v:.2f}%" for v in df_ci["SOV"]]

            fig.add_trace(go.Scatter(
                x=df_ci["TRP"],
                y=df_ci[ycol],
                mode="markers",
                name=f"{name} • Predicted pts",
                marker=dict(color=color, size=6, symbol="circle"),
                text=hov,
                hovertemplate=('%{text}<br>TRP: %{x}<br>DTB_pred: %{y}<extra></extra>') if hov else None,
                showlegend=False
            ))

        # --- ФАКТИЧЕСКИЕ ТОЧКИ (если есть actual_df) ---
        if isinstance(actual_df, pd.DataFrame) and "metric_abs_analytics" in actual_df.columns:
            if "flight" in actual_df.columns:
                da = actual_df[actual_df["flight"] == name].copy()
            else:
                da = actual_df.copy()

            for c in ["TRP", "metric_abs_analytics", "mde_abs"]:
                if c in da.columns:
                    da[c] = pd.to_numeric(da[c], errors="coerce")
            da = da.dropna(subset=["TRP", "metric_abs_analytics"])

            if len(da):
                has_err = ("mde_abs" in da.columns) and bool(da["mde_abs"].notna().any())
                err_arr = da["mde_abs"].to_numpy() if has_err else None

                fig.add_trace(go.Scatter(
                    x=da["TRP"],
                    y=da["metric_abs_analytics"],
                    mode="markers",
                    name=f"{name} • Actual",
                    marker=dict(color=color, size=8, symbol="diamond"),
                    error_y=dict(
                        type="data",
                        array=err_arr,
                        visible=bool(has_err)
                    )
                ))

    # Оформление
    title_txt = title or (items[0][0] + " — DTB" if items else "DTB")
    fig.update_layout(
        title=dict(text=title_txt, x=0.01, xanchor="left"),
        xaxis_title="TRP",
        yaxis_title="DTB",
        template="plotly_white",
        height=650,
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(0,0,0,0)")
    )

    # Вывод
    if streamlit_container is not None:
        streamlit_container.plotly_chart(fig, use_container_width=True)
    else:
        fig.show()

    return fig


# ======================================
# UI
# ======================================
st.title("pROMIser")
st.title("Предсказание эффективности флайта")


VERTICALS = ["Goods", "Realty", "Travel", "Services", "Transport", "Vacancies"]

CLOSEST_START_RK = {
    'CC-Test': [
        'Avito_Goods_gc-24-01-02-TRX-Buyer-Protection-T1-C2C',
        'Avito_Goods_gc-24-09-AvitoMall+T&S-T1'
    ],
    'HL-Construction': [
        'Avito_Goods_gc-24-06-H&L-T1-Construction-SALE',
        'Avito_Goods_gc-24-04-HL-Construction-XY-T1-C2C',
        'Avito_Goods_gc-25-04-H&L-T1-Construction'
    ],
    'HL-Furniture': ['Avito_Goods_gc-24-10-HL-Furniture-Feature-C2C'],
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
        'Avito_Goods_gc-24-12-EL&LS-SALE-T1'
    ],
    'LS': [
        'Avito_Goods_gc-24-09-AvitoMall+T&S-T1',
        'Avito_Goods_gc-24-06-СС-SALE-Federal',
        'Avito_Goods_gc-24-12-EL&LS-SALE-T1'
    ],
    'Jobs': [
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
    'LTR': ['Avito_RE_re-24-08-LTR-авг-сент-T1-С2С'],
    'Services': [
        'Avito_Services_se_24-08-CROSS_MR&TR-T1-С2С',
        'Avito_Services_se_24-06-HH-T1-С2С'
    ],
    'Auto': [
        'Avito_Auto_au-24-09-SL-Select-T1-C2C',
        'Avito_Auto_au-24-10-NCB-Buyers-T1-C2C'
    ]
}
ALL_CATEGORIES = list(CLOSEST_START_RK.keys())

VERTICAL_TO_CATEGORIES = {
    "Goods":     ["CC-Test", "HL-Construction", "HL-Furniture", "SP", "SP-Tires", "Sale", "EL", "LS"],
    "Realty":    ["RRE", "LTR"],
    "Services":  ["Services"],
    "Transport": ["Auto"],
    "Vacancies": ["Jobs"],
    "Travel":    ["STR"]
}

VERTICAL_LOGCATS = {
    "Goods": [
        "Any", "Goods.Business", "Goods.Fashion", "Goods.Sports", "Goods.SpareParts",
        "Goods.GoodsForPets", "Goods.HomeAndGarden", "Goods.Telecom", "Goods.AudioVideo",
        "Goods.HealthAndBeauty", "Goods.SparePartsServices", "Goods.CommtransSpareParts",
        "Goods.Hobby", "Goods.GoodsForChildren", "Goods.ConstructionRenovation",
        "Goods.InformationTechnology", "Goods.Animals", "Goods.Furniture",
        "Goods.DomesticAppliances", "Goods.TiresAndWheels", "Goods.Food",
    ],
    "Realty": ["Any", "Realty.Commercial", "Realty.Foreign", "Realty.LongRent", "Realty.NewDevelopments", "Realty.Other", "Realty.SecondarySell", "Realty.Suburban"],
    "Services": ["Any", "Services.Appliances", "Services.Business", "Services.EventsAndEntertaiments", "Services.HealthAndBeauty", "Services.Household", "Services.Machinery", "Services.MajorRepair", "Services.MinorRepair", "Services.Other", "Services.Training", "Services.TransportationAndDelivery"],
    "Transport": ["Any", "Transport.CarRentals", "Transport.MachineryRentals", "Transport.Moto", "Transport.NewCars", "Transport.UsedCars", "Transport.UsedMachinery", "Transport.Water"],
    "Vacancies": ["Any", "Vacancies.ManualAndLinear", "Vacancies.Office", "Vacancies.Other", "Vacancies.TaxiRentals", "Vacancies.TaxiTransportLogistic"],
    "Travel": ["Any", "Travel.Rent", "Travel.ShortRent"],
}

# --- 2) Инициализация состояния ---
if "campaign_input" not in st.session_state:
    st.session_state["campaign_input"] = {}

# Утилиты
all_flights = sorted(set(chain.from_iterable(CLOSEST_START_RK.values())))

def categories_for_vertical(vertical: str):
    cats = VERTICAL_TO_CATEGORIES.get(vertical)
    return cats if cats else ALL_CATEGORIES

# --- 3) Выбор режима ---
mode = st.radio(
    "Что проверяем?",
    options=("Old", "New"),
    format_func=lambda x: "Историческая (Old)" if x == "Old" else "Новая (New)",
    horizontal=True,
    key="campaign_mode",
)

st.divider()

# --- 4) Форма параметров (форма только сохраняет параметры) ---
with st.form("campaign_form", clear_on_submit=False):
    if mode == "Old":
        st.caption("Для исторической кампании укажи полное название флайта")
        old_name = st.text_input("Полное название флайта")
        st.selectbox("Подсказка (по справочнику)", ["—"] + all_flights, index=0, key="_hint_flight")
        submitted = st.form_submit_button("Сохранить параметры")
        if submitted:
            value = old_name.strip() or (st.session_state.get("_hint_flight") if st.session_state.get("_hint_flight") != "—" else "")
            if value:
                st.session_state["campaign_input"] = {
                    "mode": "Old",
                    "flight_name": value,
                    "vertical": None,
                    "start_date": None,
                    "end_date": None,
                    "logic_categories": [],
                    "category": None,
                    "trp": None,
                    "sov": None,
                }
                st.success("Параметры сохранены")
            else:
                st.warning("Укажи полное название флайта или выбери из подсказок.")
    else:  # New
        col1, col2 = st.columns(2)
        with col1:
            vertical   = st.selectbox("Вертикаль", VERTICALS)
            start_date = st.date_input("Дата начала", value=date.today())
            end_date   = st.date_input("Дата окончания", value=date.today())
            trp        = st.number_input("TRP", min_value=0.0, step=1.0, help="Total Rating Points за период")
            opm        = st.number_input("OPM", min_value=0.0, step=1.0, help="Значение метрики OPM из претестов")
        with col2:
            allowed_categories = categories_for_vertical(vertical)
            category = st.selectbox("Категория", allowed_categories)
            logcats_options = VERTICAL_LOGCATS.get(vertical, [])
            logic    = st.multiselect("Логические категории", logcats_options)
            sov      = st.number_input("SOV, %", min_value=0.0, max_value=100.0, step=0.1, help="Share of Voice за период")
            consideration      = st.number_input("Consideration", min_value=0.0, max_value=100.0, step=1.0, help="Захотели воспользоваться Авито")

        submitted = st.form_submit_button("Сохранить параметры")
        if submitted:
            if end_date < start_date:
                st.warning("'Дата окончания' не может быть раньше 'Дата начала'.")
            elif not category:
                st.warning("Выбери категорию.")
            else:
                st.session_state["campaign_input"] = {
                    "mode": "New",
                    "flight_name": "NEW-Flight",
                    "vertical": str(vertical),
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "logic_categories": list(logic),
                    "category": category,
                    "trp": int(trp) if trp is not None else None,
                    "sov": int(sov) if sov is not None else None,
                    "opm": int(opm) if opm is not None else None,
                    "consideration": int(consideration) if consideration is not None else None
                }
                st.success("Параметры сохранены")

# Отдельная кнопка запуска расчёта (строго по ней)
run_click = st.button("Предсказание", type="primary")

# ======================================
# Бизнес-логика (СТРОГО по кнопке)
# ======================================
if not run_click:
    st.stop()

cfg = st.session_state.get("campaign_input") or {}
if not cfg:
    st.warning("Сначала заполните форму и нажмите «Сохранить параметры».")
    st.stop()

with st.spinner("Считаем предсказание..."):
    # Подключения создаём только здесь — после клика
    c_engine, t_engine = connect_dwh(password)
    df = pd.read_excel('/Users/asekorneev/Documents/Work projects/Код/df_for_pROMIser.xlsx')
    with open('/Users/asekorneev/Documents/Work projects/Код/flight_trp_dict.pkl', 'rb') as f:
        flight_trp_dict = pickle.load(f)

    if cfg["mode"] == "New":
        # Вспомогательный предиктор для подготовительных вычислений
        base_predictor = pROMIser_t(c_engine, t_engine)

        sd = pd.to_datetime(cfg["start_date"]).date()
        ed = pd.to_datetime(cfg["end_date"]).date()
        vertical   = str(cfg["vertical"])
        logic_list = list(map(str, cfg.get("logic_categories", []) or []))

        base_dtb = base_predictor.get_dtb_base_for_flight(vertical, logic_list, sd, ed)

        # one-row датасет
        data = pd.DataFrame([{
            "flight": cfg.get("flight_name") or "NEW-Flight",
            "vertical": vertical,
            "date_start": sd,
            "date_end": ed,
            "logical_category": (logic_list[0] if logic_list else None),
            "category": cfg["category"],
            "TRP": cfg.get("trp"),
            "SOV": cfg.get("sov"),
            "metric_abs_analytics": np.nan,
            "mde_abs": np.nan,
            "base_dtb": base_dtb,
        }])

        campaigns = [data.loc[0, "flight"]]
        predictor = pROMIser_t(c_engine, t_engine, data_to_predict=data, campaigns_list=campaigns, flight_trp_dict=flight_trp_dict, df=df, cons=consideration, opm=opm)

        predictor.add_to_trp_dict(data)
        res = predictor.predict_dtb(campaigns_type="New", show_res=False)
        ci_dict = predictor.confidence_intervals_for_prediction(data, n_bootstrap=100, ci=(2.5, 97.5))

    else:  # Old
        old_flight = cfg.get("flight_name")
        predictor = pROMIser_t(c_engine, t_engine, campaigns_list=[old_flight], flight_trp_dict=flight_trp_dict, df=df)
        res = predictor.predict_dtb(campaigns_type="Old", show_res=False)
        ci_dict = predictor.confidence_intervals_for_prediction(predictor.df, n_bootstrap=100, ci=(2.5, 97.5))

visualize_ci_prediction(ci_dict, predictor.df, streamlit_container=st, title=f"{cfg.get('flight_name')} — DTB")
st.table(ci_dict)
st.metric("DTB", res['plan'][0])
