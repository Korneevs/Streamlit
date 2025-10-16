# Пакеты.
import pandas as pd
import os
import inspect
from collections import defaultdict
from copy import copy
from hashlib import blake2b

import sys
sys.path.append('../Lib')
import bxtools.vertica as vertica
import media.utils as utils


# Список базовых метрик для PLATVA и где они находятся (название файла).
# Используется дальше в get_full_metric_table.
global metric_to_sql 

metric_to_sql = {
            'buyers': 'platva_main_metrics',
            'contacts': 'platva_main_metrics',
            'DLU': 'platva_main_metrics',
            'DTB': 'platva_main_metrics',
            'iv': 'platva_main_metrics'
}


class PLATVA_metric_dataset():
    """
    В данном классе выгружаются датасеты с метриками в разрезе 
        пользователей, дат и группы (Тест / Контроль / Авито) для каждого разреза.
    """

    def __init__(self, v_engine) -> None:
        """
        Инициализация класса.
        :param v_engine: Подключение к Vertica.
        """
        self.raw_dataframe = False
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_columns', None)
        
        # Подключаемся к Vertica.
        self.vertica_engine = v_engine
        
        
    def get_hashed_table_name(self, json_params, slice_dict) -> str:
        """
        Генерирует уникальное имя таблицы на основе параметров эксперимента.
        :param json_params: Параметры эксперимента.
        :param slice_dict: Параметры среза.
        :return: Хешированное название таблицы, например, 's02ff9deade7dbe0a9...'.
        """
        key = str(json_params) + str(slice_dict)
        key = key.replace(' ', '').replace('\n', '').replace('\t', '')
        text = bytes(key, 'utf-8')
        h = blake2b(digest_size=10)
        h.update(text)
        hashed_name = 's' + h.hexdigest()

        return hashed_name

    
    def load_sql_template(self, file_name, is_custom) -> str:
        """
        Загружает SQL-шаблон.
        :param file_name: Название SQL-файла.
        :param is_custom: Флаг для кастомной метрики. Если True, то файл выгружается из рабочей директории, 
            если False - из директории, где находится фреймворк.
        :return: SQL-шаблон в виде строки.
        """
        if is_custom:
            # Ищет путь в рабочей директории.
            path = os.getcwd() + '/' + file_name
        else:   
            # Ищет путь в директории с фреймворком.
            path = "/".join(inspect.getfile(self.__class__).split('/')[:-1])
            path = path + f'/SQL/{file_name}.sql'
        
        return utils.read_sql_file(os.path.abspath(path))

    
    def fill_not_filled_query(self, is_custom, curr_metrics, metric_table_path, 
                              metric_table_name, flight_params, slice_dict, metrics_current_slice) -> str:
        """
        Формирует SQL-запрос, заполняя шаблон параметрами.

        1. Загружает SQL-шаблон через (из metric_table_path).
        2. Подставляет в шаблон параметры (metrics, log_cat_str и т.д.).
        3. Заполняет нулями все метрики, которые: 
            есть в общем списке метрик для текущего разреза metrics_current_slice и нет в текущих метриках curr_metrics, 
            которые рассчитываются в запросе по пути metric_table_path.
        4. Генерирует финальный SQL-запрос.

        :param is_custom: Используется ли кастомный SQL-файл.
        :param metric_table_path: Название SQL-файла.
        :param curr_metrics: Метрики, которые рассчитываются в файле из metric_table_path.
        :param metric_table_name: Название временной таблицы.
        :param flight_params: Параметры кампании.
        :param slice_dict: Параметры среза.
        :param metrics_current_slice: Метрики текущего разреза.
        :return: Финальный SQL-запрос.
        """

        # Загружаем SQL-шаблон.
        sql_query = self.load_sql_template(metric_table_path, is_custom)

        # Формируем SQL-запрос, добавляя нулевые значения для отсутствующих метрик.
        other_metrics = set(metrics_current_slice) - set(curr_metrics)
        standart_params = {
            # Логические категории.
            'log_cat_str': str(slice_dict['logical_category'])[1:-1],
            
            # Остальные метрики заполняются нулями.
            'other_metrics': "\n".join(f"0 AS {m}," for m in other_metrics)
        }
        
        # Параметры и генерация запроса.
        all_params = {**standart_params, **flight_params, **slice_dict}
        sql_query = utils.fill_query(sql_query, all_params)
        sql_query = self.add_saving_for_sql(sql_query, metric_table_name)
        
        return sql_query

    
    def unite_all_metrics_dataset(self, metric_tables, metrics) -> None:
        """
        Объединяет все временные таблицы метрик в одну (full_metric_table).

        1. Формирует SQL-запрос, объединяя все metric_tables с нужными metrics.
        2. Выполняет SQL-запрос.

        :param metric_tables: Список временных таблиц.
        :param metrics: Список метрик.
        :return: Ничего.
        """

        # Формируем SQL-запрос, объединяющий все таблицы.
        sql_queries = [
            f"SELECT user_id, dt, {', '.join(metrics)} FROM {table}" for table in metric_tables
        ]
        final_sql = "\nUNION ALL\n".join(sql_queries)

        # Сохраняем объединённый датасет.
        final_sql = self.add_saving_for_sql(final_sql, 'full_metric_table')

        # Выполняем SQL.
        self.execute_query(final_sql)

        
    def get_full_metric_table(self, table_name, flight_params, slice_dict, metrics_current_slice) -> None:
        """
        Создаёт полную таблицу метрик для анализа.

        1. Определяет, какие метрики нужно получить и из каких SQL-файлов.
        2. Формирует SQL-запросы для каждой группы метрик.
        3. Выполняет SQL-запросы, создавая временные таблицы.
        4. Объединяет все таблицы в одну (unite_all_metrics_dataset).

        :param table_name: Название итоговой таблицы.
        :param flight_params: Параметры рекламной кампании.
        :param slice_dict: Параметры среза данных.
        :param metrics_current_slice: Метрики текущего разреза.
        :return: Ничего.
        """

        # Проверяем, есть ли кастомные SQL-файлы с метриками.
        custom_metric_sql = slice_dict.get('metric_dict', {})

        # Убеждаемся, что нет конфликта между стандартными и кастомными метриками.
        if len(set(metric_to_sql.keys()) & set(custom_metric_sql.keys())) != 0:
            raise ValueError(f'Название кастомной метрики входит в базовые метрики: {list(metric_to_sql.keys())}!')

        # Определяем уникальные файлы с метриками.
        standart_tables = set(metric_to_sql.values())
        custom_tables = set(custom_metric_sql.values())

        # Создаём словарь {название SQL-файла -> список метрик, которые он содержит}.
        table_to_metrics = defaultdict(lambda: list())
        for metric, table in metric_to_sql.items():
            table_to_metrics[table].append(metric)
            
        for metric, table in custom_metric_sql.items():
            table_to_metrics[table].append(metric)
        
        # Создаём временные таблицы для каждой группы метрик.
        metric_tables = []
        index = 0
        for table in standart_tables:
            metric_table_name = f'table_{index}'
            index += 1
            # Генерируем SQL-запрос для создания таблицы с метриками.
            sql = self.fill_not_filled_query(False, table_to_metrics[table], 
                                             table, metric_table_name, flight_params, slice_dict, metrics_current_slice)
            
            # Добавляем таблицу в список.
            metric_tables.append(metric_table_name)
            self.execute_query(sql)
            
        for table in custom_tables:
            metric_table_name = f'table_{index}'
            index += 1
            # Генерируем SQL-запрос для создания таблицы с метриками.
            sql = self.fill_not_filled_query(True, table_to_metrics[table],
                                             table, metric_table_name, flight_params, slice_dict, metrics_current_slice)
            
            # Добавляем таблицу в список.
            metric_tables.append(metric_table_name)
            self.execute_query(sql)
        
        # Объединяем датасеты.
        self.unite_all_metrics_dataset(metric_tables, metrics_current_slice)


    def add_saving_for_sql(self, sql_query, table_name) -> str:
        """
        Создаёт временную таблицу для хранения результата.

        1. Удаляет таблицу, если она уже существует.
        2. Создаёт новую таблицу с нужными данными.

        :param sql_query: SQL-запрос для создания таблицы.
        :param table_name: Название таблицы.
        :return: Финальный SQL-запрос.
        """

        return f"""
        DROP TABLE IF EXISTS {table_name};
        CREATE LOCAL TEMP TABLE {table_name} ON COMMIT PRESERVE ROWS AS
        /*+DIRECT*/
        (
            {sql_query}
        );
        """

    
    def execute_query(self, sql) -> None:
        """
        Выполняет SQL-запрос в Vertica.

        1. Делит SQL на отдельные команды (если несколько команд в одном запросе).
        2. Запускает выполнение SQL в Vertica.

        :param sql: SQL-запрос.
        :return: Ничего.
        """
        
        for query in sql.strip().split(';'):
            if query.strip():  # Игнорируем пустые строки.
                self.vertica_engine.execute(query.strip())

    
    def get_filled_sql_dataset_query(self, table_name, flight_params, slice_dict, metrics_current_slice) -> str:
        """
        Генерирует финальный SQL-запрос для создания датасета.

        1. Вызывает get_full_metric_table() для сбора всех метрик.
        2. Загружает SQL-шаблон (metric_table.sql).
        3. Подставляет в SQL список метрик и таблицу.

        :param table_name: Название итоговой таблицы.
        :param flight_params: Параметры кампании.
        :param slice_dict: Параметры среза.
        :param metrics_current_slice: Метрики текущего разреза.
        :return: Финальный SQL-запрос.
        """
            
        # Получаем полную таблицу для сбора всех метрик.
        self.get_full_metric_table(table_name, flight_params, slice_dict, metrics_current_slice)
        
        # Формируем шаблон из файла metric_table.sql.
        sql_query = self.load_sql_template('metric_table', is_custom=False)
        
        # Суммируем все метрики из текущего разреза.
        selected_metrics = ",\n".join([
            f"COALESCE(SUM({m}), 0) AS {m}"
            for m in metrics_current_slice
        ])

        return utils.fill_query(sql_query, {
            'tv_nontv_users_table': slice_dict.get('custom_user_table', 'tv_nontv_users'),
            'table_name': table_name,
            'sum_metrics': selected_metrics
        })
    

    def create_dataset(self, table_name, flight_params, slice_dict, metrics_current_slice) -> None:
        """
        Создаёт датасет в Vertica.

        1. Генерирует SQL-запрос для создания таблицы (get_filled_sql_dataset_query).
        2. Выполняет SQL-запрос (execute_query).

        :param table_name: Название таблицы.
        :param flight_params: Параметры рекламной кампании.
        :param slice_dict: Параметры среза.
        :param metrics_current_slice: Метрики текущего разреза.
        :return: Ничего.
        """

        # Генерируем SQL-запрос для получения датасета.
        sql_query = self.get_filled_sql_dataset_query(table_name, flight_params, slice_dict, metrics_current_slice)
        
        # Выполняем SQL-запрос в Vertica.
        self.execute_query(sql_query)

    
    def get_slice_dataset(self, flight_params, slice_dict, metrics_current_slice) -> str:
        """
        Создаёт и возвращает имя таблицы для заданного среза данных.

        :param flight_params: Параметры рекламной кампании.
        :param slice_dict: Параметры среза данных.
        :param metrics_current_slice: Метрики текущего разреза.
        :return: Строка с названием таблицы, например, 's02ff9deade7dbe0a9...'.
        """

        # Генерируем уникальное имя таблицы на основе параметров.
        table_name = self.get_hashed_table_name(flight_params, slice_dict)

        # Создаём таблицу с данными.
        self.create_dataset(table_name, flight_params, slice_dict, metrics_current_slice)

        # Возвращаем название таблицы.
        return table_name
    

    def get_final_datasets(self, flight_params, slices_json) -> dict:
        """
        Создаёт финальные датасеты для всех срезов данных из slices_json.

        1. Для каждого slice_name вызывает get_slice_dataset().
        2. Сохраняет название итоговой таблицы и список метрик.

        :param flight_params: Параметры рекламной кампании.
        :param slices_json: Список срезов.
        :return: Словарь с результатами.
        """
    
        # Сюда будем складывать название таблицы с данными по срезу, а также список метрик.
        dataset_dict = {}
        
        # Проверка на наличие всех метрик из разрезов в 'metrics' Json main_params.
        metrics_in_slices = []
        for slice_dict in slices_json.values():
            if 'metric_dict' in slice_dict:
                metrics_in_slices.extend(list(slice_dict['metric_dict'].keys()))
        
        bad_metrics = [
            metric for metric in metrics_in_slices if metric not in flight_params['metrics']
        ]
        
        if len(bad_metrics) != 0:
            raise ValueError(f"Метрики {bad_metrics} должны быть указаны по ключу 'metrics' в Json main_params!")

        # Теперь бегаем по срезам для расчета таблиц.
        for slice_name, slice_dict in slices_json.items():
            # Список метрик для данного среза: базовые + метрики для конкретного среза.
            metrics_current_slice = list(
                (set(flight_params['metrics']) & set(metric_to_sql.keys())) | set(slice_dict.get('metric_dict', {}))
            )
            
            if len(metrics_current_slice) == 0:
                raise ValueError("Должна быть хотя бы одна метрика для конкретного разреза!")
            
            # Создаём таблицу для текущего среза.
            slice_dataset_name = self.get_slice_dataset(flight_params, slice_dict, metrics_current_slice)
            
            # Сохраняем название таблицы и метрики для текущего среза.
            dataset_dict[slice_name] = {
                'final_dataset': slice_dataset_name,
                'metrics': metrics_current_slice
            }

        return dataset_dict
    