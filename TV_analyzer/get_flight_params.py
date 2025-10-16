import os
import inspect
import pandas as pd
from IPython.display import display, HTML
import media.utils as utils


def get_flight_params(main_params: dict, vertica_engine, show_params: bool = False) -> dict:
    """
    Обновляет словарь main_params с параметрами кампании.
    """
    # Если указан flight_name_viz, подтягиваем недостающие параметры через SQL.
    if main_params.get('flight_name_viz'):
        sql_file_path = os.path.abspath(
            os.path.join(
                os.path.dirname(inspect.getfile(inspect.currentframe())),
                'SQL',
                'flight_params.sql'
            )
        )
        sql_query = utils.read_sql_file(sql_file_path) \
            .format(flight_name_viz=main_params['flight_name_viz'])
        sql_query = sql_query
        
        result_df = vertica_engine.select(sql_query)
        if result_df.empty:
            raise ValueError("Такого названия РК в визуализаторе нет!")
        
        # Извлекаем первую строку результата.
        row = result_df.iloc[0]
        
        for param in result_df.columns:
            # Если данный параметр не указан в main_params, то указываем его из результата запроса result_df.
            if param not in main_params:
                value = row[param]
                
                # Если такого параметра нет (например, P&S его не указали или какая-то бага), то мы выкидываем ошибку.
                if value is None:
                    raise ValueError(f"Данных по этой РК сейчас нет, поэтому нужно уточнять параметры у медиа-менеджеров!")
                    
                # Если это ID роликов - преобразуем в список из чисел.
                if param == 'rk_clip_id':
                    value = list(map(int, result_df['rk_clip_id'][0].split(', ')))
                elif param.endswith('_date'):
                    # Приводим значение к строке в формате 'YYYY-MM-DD'.
                    value = pd.to_datetime(value).strftime('%Y-%m-%d')
                main_params[param] = value

    # Устанавливаем значения по умолчанию для регионов всегда.
    defaults = {
        'test regions': ['Any'],
        'control regions': [],
        'exclude regions': []
    }
    
    for param, value in defaults.items():
        main_params[param] = value

    # Выводим обновленные параметры в виде транспонированной HTML‑таблицы, если задан show_params.
    if show_params:
        df = pd.DataFrame([main_params]).T
        df.columns = ['Value']
        html_table = df.to_html(classes="table table-bordered table-striped", border=0)
        display(HTML(html_table))

    return main_params
