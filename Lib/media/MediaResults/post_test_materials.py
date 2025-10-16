import re
import datetime as dt
import pandas as pd


# цвета для колонок
COLORS = {
    "GREEN":  {"red": 236/255, "green": 244/255, "blue": 232/255},
    "BLUE":   {"red": 224/255, "green": 241/255, "blue": 253/255},
    "PINK":   {"red": 254/255, "green": 233/255, "blue": 242/255},
    "YELLOW": {"red": 255/255, "green": 248/255, "blue": 226/255},
    "ORANGE": {"red": 255/255, "green": 238/255, "blue": 205/255},
    "LILAC":  {"red": 241/255, "green": 236/255, "blue": 253/255},
    "WHITE":  {"red": 1,       "green": 1,       "blue": 1},
    "GREY":   {"red": 240/255, "green": 240/255, "blue": 240/255},
}

# порядок цветов для шапок метрик
GROUP_COLORS = [
    COLORS["BLUE"], COLORS["PINK"], COLORS["YELLOW"], COLORS["LILAC"],
    COLORS["ORANGE"], COLORS["GREEN"], COLORS["BLUE"], COLORS["PINK"],
    COLORS["YELLOW"]
]

MIN_METRIC_WIDTH = 115        # минимальная ширина колонки, px


class MediaPostTest:
    """
    Класс для создания Google-таблиц с пост тестами.

    googlya : см. Lib/Google/googlya.py
    folder_tags : список имён папок, куда пишем файл
    file_name   : имя файла-таблицы
    raw_by_channel : dict{channel: DataFrame} | None
        SQL-срезы по каналам (TV, OOH, Digital) 
    """

    # ------------ константы --------------------------------
    CHANNEL_SHEETS = {
        "TV":               "Post Test TV",
        "OOH":              "Post Test OOH",
        "Digital":          "Post Test Digital",
        "Internet Banners": "Post Test Internet Banners",
    }

    # метрика на русском, имя столбца на английском
    METRICS = [
        ("Знание",                        "ad_recall"),
        ("Считываемость бренда",          "brand_awareness"),
        ("Считываемость основной идеи",   "idea_awareness"),
        ("Считываемость вертикали",       "vertical_awareness"),
        ("Интерес к рекламе",             "ad_interest"),
        ("Влияние на доверие бренду",     "trust_impact"),
        ("Желание воспользоваться",       "desire_to_use"),
        ("KPI",                           "kpi"),
    ]

    # метрики для колонок с Базой
    METRICS_WITH_BASE = {"Знание", "Считываемость бренда", "KPI"}
    KPI_METRIC = "KPI"
    CITY_SUFFIXES = ["", " Мск", " СПб", " Города 100к+"]

    BASE_RU = [
        "Название флайта", "ID Креатив", "Компания", "Вертикаль",
        "Дата начала", "Дата окончания", "Материалы", "Cтатус Креатива", "TRP", "Общий имидж"
    ]
    BASE_EN = [
        "flight_name", "creative_id", "company", "vertical",
        "date_start", "date_end", "materials", "creative_status", "post_trp", "image"
    ]

    # ------------ инициализация ------------------------------
    def __init__(self, googlya, folder_tags, file_name, raw_by_channel=None):
        self.googlya = googlya
        self.folder_tags = folder_tags
        self.file_name = file_name
        self.raw = raw_by_channel or {}

    # можно задать SQL-данные позже
    def set_raw(self, raw_by_channel):
        """Сохраняем или заменяем SQL-данные по каналам."""
        self.raw = raw_by_channel
        
    def _ensure_file(self):
        return self.googlya.create_googlesheets(self.folder_tags, self.file_name)

    # ------------ свойства ------------------------------
    @property
    def RUS_COLUMNS(self):
        """
        :return: Полный перечень колонок в фиксированном порядке на русском языке.
        """
        cols = self.BASE_RU.copy()
        for ru, _ in self.METRICS:
            # Пример: KPI -> KPI Мск: База 
            if ru == self.KPI_METRIC:
                cols += [f"{ru}{s}: База" if s else f"{ru}: База"
                         for s in self.CITY_SUFFIXES]
                continue
            for suf in self.CITY_SUFFIXES:
                live = f"{ru}{suf}"
                cols.append(live)
                if ru in self.METRICS_WITH_BASE:
                    cols.append(f"{live}: База" if suf else f"{ru}: База")
        return cols

    @property
    def RUS2ENG(self):
        """
        :return: Словарь c переводом из русского названия колонок в английское
        """
        m = dict(zip(self.BASE_RU, self.BASE_EN))
        city_pairs = [
            ("", ""), (" Мск", "_msk"), (" СПб", "_spb"), (" Города 100к+", "_100k_cities")
        ]
        for ru, en in self.METRICS:
            for r_suf, e_suf in city_pairs:
                ru_live = f"{ru}{r_suf}".strip()
                en_live = f"{en}{e_suf}"
                if ru != self.KPI_METRIC:
                    m[ru_live] = en_live
                if ru in self.METRICS_WITH_BASE or ru == self.KPI_METRIC:
                    ru_base = f"{ru_live}: База" if r_suf else f"{ru}: База"
                    m[ru_base] = f"{en_live}_base"
        return m

    # ------------ служебные методы ------------------------------
    @staticmethod
    def _clean_header(text):
        """Удаляем перевод строки и лишние пробелы из названия колонки"""
        return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    
    @staticmethod
    def _display(col: str) -> str:
        """
        Форматирует название колонки для шапки.

        'Знание Мск'        → 'Знание\\nМск'
        'Знание: База'      → 'Знание:\\nБаза'
        'Знание СПб: База'  → 'Знание\\nСПб:\\nБаза'
        """
        for tok in (" Города 100к+", " СПб", " Мск"):
            if tok in col:
                col = col.replace(tok, f"\n{tok.strip()}")
                break
        return col.replace(": База", ":\nБаза")

    @property
    def DISPLAY_COLUMNS(self):
        """
        Список колоночных заголовков со вставленными переносами строк.
        """
        return [self._display(c) for c in self.RUS_COLUMNS]

    # ------------ работа с столбцом Нормы ------------------------------
    def _norm_row(self):
        """Создаём строку «Норма»: прочёрки и пустые значения."""
        row = {c: "-" for c in self.RUS_COLUMNS}
        for ru, _ in self.METRICS:
            if ru == self.KPI_METRIC:
                continue
            for suf in self.CITY_SUFFIXES:
                row[f"{ru}{suf}"] = ""
        row["Название флайта"] = "Норма"
        return pd.DataFrame([row], columns=self.RUS_COLUMNS)
    
    def _mk_norm_cols(self, df_eng):
        """
        Преобразование строки Норма в *_norm (англ. формат)
        """
        norm = df_eng[df_eng["flight_name"] == "Норма"].iloc[0]
        df_eng = df_eng[df_eng["flight_name"] != "Норма"].reset_index(drop=True)
        for ru, en in self.METRICS:
            if ru == self.KPI_METRIC:
                continue
            df_eng[f"{en}_norm"] = norm.get(en, "")
        return df_eng
    
    def _format_sheet(self, ss_id, sheet_id, n_rows, n_cols):
        """
        Задаём формат листа (banding, freeze, автоширина, заливка шапки,
        формат процентов / чисел).

        Все requests собираются в один batchUpdate
        """
        req = []

        # чередование цвета для строк 
        req.append({
            "addBanding": {
                "bandedRange": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 0,
                        "endRowIndex": n_rows,
                        "startColumnIndex": 0,
                        "endColumnIndex": n_cols,
                    },
                    "rowProperties": {
                        "headerColor":     COLORS["GREEN"],
                        "firstBandColor":  COLORS["WHITE"],
                        "secondBandColor": COLORS["GREY"],
                    },
                }
            }
        })

         # фиксируем шапку 
        req.append({
            "updateSheetProperties": {
                "properties": {
                    "sheetId": sheet_id,
                    "gridProperties": {"frozenRowCount": 1},
                },
                "fields": "gridProperties.frozenRowCount",
            }
        })

        # общий формат: WRAP
        req.append({
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0,
                    "endRowIndex": n_rows,
                    "startColumnIndex": 0,
                    "endColumnIndex": n_cols,
                },
                "cell": {"userEnteredFormat": {
                    "horizontalAlignment": "LEFT",
                    "wrapStrategy": "WRAP",
                }},
                "fields": "userEnteredFormat(horizontalAlignment,wrapStrategy)",
            }
        })

        # исключение: первая колонка («Название флайта») — CLIP
        req.append({
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0,
                    "endRowIndex": n_rows,
                    "startColumnIndex": 0,
                    "endColumnIndex": 1,
                },
                "cell": {"userEnteredFormat": {
                    "wrapStrategy": "CLIP"
                }},
                "fields": "userEnteredFormat.wrapStrategy",
            }
        })

        # auto-resize
        req.append({
            "autoResizeDimensions": {
                "dimensions": {
                    "sheetId": sheet_id,
                    "dimension": "COLUMNS",
                    "startIndex": 0,
                    "endIndex": n_cols,
                }
            }
        })

        # min-width для метрик
        req.append({
            "updateDimensionProperties": {
                "range": {
                    "sheetId": sheet_id,
                    "dimension": "COLUMNS",
                    "startIndex": 1,
                    "endIndex": n_cols,
                },
                "properties": {"pixelSize": MIN_METRIC_WIDTH},
                "fields": "pixelSize",
            }
        })
        
        # italic «Норма» 
        req.append({
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 1,
                    "endRowIndex": 2,
                    "startColumnIndex": 0,
                    "endColumnIndex": 1,
                },
                "cell": {"userEnteredFormat": {"textFormat": {"italic": True}}},
                "fields": "userEnteredFormat.textFormat.italic",
            }
        })

        # шапка (значения + wrap)
        header_cells = [{
            "userEnteredValue": {"stringValue": h},
            "userEnteredFormat": {
                "horizontalAlignment": "LEFT",
                "verticalAlignment":   "MIDDLE",
                "wrapStrategy":        "WRAP",
            },
        } for h in self.DISPLAY_COLUMNS]

        req.append({
            "updateCells": {
                "rows": [{"values": header_cells}],
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0,
                    "endRowIndex": 1,
                    "startColumnIndex": 0,
                    "endColumnIndex": n_cols,
                },
                "fields": "userEnteredValue,userEnteredFormat"
            }
        })

        # цвет блоков 
        group_start = len(self.BASE_RU)
        for idx, (ru, _) in enumerate(self.METRICS):
            width = 4 if ru == self.KPI_METRIC else 4 + (4 if ru in self.METRICS_WITH_BASE else 0)
            req.append({
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 0,
                        "endRowIndex": 1,
                        "startColumnIndex": group_start,
                        "endColumnIndex": group_start + width,
                    },
                    "cell": {"userEnteredFormat": {
                        "backgroundColor": GROUP_COLORS[idx]
                    }},
                    "fields": "userEnteredFormat.backgroundColor",
                }
            })
            group_start += width
            
            
        # формат процентов / чисел 
        formats = []
        start_col = len(self.BASE_RU)
        for idx, col_name in enumerate(self.RUS_COLUMNS[start_col:], start=start_col):
            if col_name.endswith(": База"):
                nf = {"type": "NUMBER",  "pattern": "0"}   # базы и KPI → число
            else:
                nf = {"type": "PERCENT", "pattern": "0%"}  # всё остальное → %
            formats.append((idx, nf))

        # превращаем в requests
        req += [{
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 1,       
                    "endRowIndex": n_rows,   
                    "startColumnIndex": idx,
                    "endColumnIndex": idx + 1
                },
                "cell": {"userEnteredFormat": {"numberFormat": nf}},
                "fields": "userEnteredFormat.numberFormat"
            }
        } for idx, nf in formats]


        # отправка
        self.googlya.sheets_service.spreadsheets() \
            .batchUpdate(spreadsheetId=ss_id, body={"requests": req}).execute()
    
    # ------------ построение DataFrame по SQL-данным ------------------------------
    def _make_df(self, channel, wave_start, wave_end):
        """
        Строим DataFrame для одного канала и волны (с русскими названиями колонок).
        """
        if channel not in self.raw:
            df = pd.DataFrame(columns=[
                "flight_name", "company", "vertical",
                "creative_id", "date_start", "date_end", "materials"
            ])
        else:
            df = self.raw[channel]

        # фильтруем по дате
        df["date_start"] = pd.to_datetime(df["date_start"]).dt.date
        df["date_end"]   = pd.to_datetime(df["date_end"]).dt.date
        df = df[df["date_start"] <= wave_end].loc[
            :, ["flight_name", "vertical", "date_start", "date_end"]
        ].copy()

        # обязательные поля
        df["company"]     = "Avito"
        for col in ["materials", "creatice_id", "timing", "post_trp", "trp_status"]:
            df[col] = ""
        df = df.sort_values(
            by=["vertical", "date_start"],
            key=lambda col: col.map(lambda x: 1 if x == "Goods" else 0) if col.name == "vertical" else col
        )
        # переименование
        df.rename(columns=dict(zip(self.BASE_EN, self.BASE_RU)), inplace=True)

        # добавляем недостающие колонки
        for col in self.RUS_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        df = df[self.RUS_COLUMNS]
        df = pd.concat([self._norm_row(), df], ignore_index=True)
        return df


    # ------------ запись в листы по каналам ------------------------------
    def rewrite_sheet(self, channel, wave_start, wave_end):
        """
        Полностью перезаписывает лист для канала.
        :return: Англоязычный DataFrame (строка «Норма» уже в *_norm)
        """
        ss_id = self.googlya.create_googlesheets(self.folder_tags, self.file_name)
        df_ru = self._make_df(channel, wave_start, wave_end)

        # заливаем данные в гугл-лист
        self.googlya.make_raw_res(df_ru,
                                  self.folder_tags + [self.file_name],
                                  self.CHANNEL_SHEETS[channel])

        # удаляем дефолтный лист
        sh = self.googlya.client.open_by_key(ss_id)
        for ws in sh.worksheets():
            if ws.title.lower() in {"sheet1", "лист1", "sheet"} and len(sh.worksheets()) > 1:
                sh.del_worksheet(ws)

        # форматирование
        ws = sh.worksheet(self.CHANNEL_SHEETS[channel])
        self._format_sheet(ss_id, ws.id, len(df_ru) + 1, df_ru.shape[1])

        return self._mk_norm_cols(df_ru.rename(columns=self.RUS2ENG))
