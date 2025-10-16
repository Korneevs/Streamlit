import os
import sys
import inspect
import datetime
import pandas as pd

# Добавляем путь к библиотекам
sys.path.append('../Lib')
import bxtools.vertica as vertica
import media.utils as utils
from media.TV_analyzer.avito_channels import avito_channels


class PlatvaTVNONTVUsersMaker:
    """
    Класс для формирования таблицы с пользователями и информацией о просмотрах ТВ.
    Вся логика создания таблиц вынесена в отдельный SQL‑файл tv_nontv_users.sql.
    """

    def __init__(self, v_engine):
        pd.set_option('display.max_columns', None)
        self.vertica_engine = v_engine

        
    def execute_query(self, sql) -> None:
        """
        Выполняет SQL-запрос в Vertica.

        SQL разбивается на отдельные команды (если их несколько в одном файле)
        и выполняется по очереди.

        :param sql: Текст SQL-запроса.
        """
        for query in sql.strip().split(';'):
            if query.strip():  # Игнорируем пустые строки.
                self.vertica_engine.execute(query.strip())

                
    def get_week_boundaries(self, flight_start_date, flight_end_date):
        """
        Вычисляет границы недели и период full_period.

        Вычисляет:
            - date_week_start: дату начала недели для flight_start_date,
            - date_week_end: дату начала недели для flight_end_date,
            - full_period: разницу в днях между date_week_end и date_week_start плюс 7 дней.

        :param flight_start_date: Дата начала кампании.
        :param flight_end_date: Дата окончания кампании.
        
        :return: Кортеж (date_week_start, date_week_end, full_period).
        """
        date_week_start = pd.to_datetime(flight_start_date) \
            .to_period('W').start_time.strftime('%Y-%m-%d')
        date_week_end = pd.to_datetime(flight_end_date) \
            .to_period('W').start_time.strftime('%Y-%m-%d')
        full_period = (pd.to_datetime(date_week_end) -
                       pd.to_datetime(date_week_start)).days + 7

        return date_week_start, date_week_end, full_period

    def create_tv_nontv_user_table(self, main_params):
        """
        Формирует итоговую таблицу tv_nontv_users.

        main_params должен содержать:
            - flight_start_date: дата начала кампании,
            - flight_end_date: дата окончания кампании,
            - rk_clip_id: список идентификаторов рекламных роликов.

        Логика:
            1. Вычисляются date_week_start, date_week_end и full_period.
            2. Загружается SQL‑файл tv_nontv_users.sql с запросами.
            3. SQL‑текст форматируется с подстановкой необходимых параметров.
            4. Выполняется сформированный SQL.

        :param main_params: Словарь с параметрами кампании.
        """
        flight_start_date = main_params['flight_start_date']
        flight_end_date = main_params['flight_end_date']
        rk_clip_ids = main_params['rk_clip_id']

        # Вычисляем границы недели и период.
        date_week_start, date_week_end, full_period = self.get_week_boundaries(
            flight_start_date, flight_end_date
        )

        # Загружаем SQL‑файл с запросами.
        sql_file_path = os.path.abspath(os.path.join(
            os.path.dirname(inspect.getfile(self.__class__)),
            'SQL',
            'tv_nontv_users.sql'
        ))
        sql_all = utils.read_sql_file(sql_file_path)

        # Приводим список роликов и каналов в строковый формат для подстановки в SQL.
        rk_clip_ids_str = str(rk_clip_ids)[1:-1]
        avito_channels_str = str(avito_channels)[1:-1].replace('\\\\', '\\')

        # Форматируем SQL с необходимыми параметрами.
        sql_all = sql_all.format(
            date_week_start=date_week_start,
            date_week_end=date_week_end,
            full_period=full_period,
            rk_ids=rk_clip_ids_str,
            channels=avito_channels_str
        )

        # Выполняем сформированный SQL.
        self.execute_query(sql_all)
