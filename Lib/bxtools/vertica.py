import pandas as pd
import os
import logging
import itertools
import vertica_python
import textwrap
import os.path as path
import time
import datetime
from log_helper import configure_logger, log_time, log_exception
from .utils import define_logger, iter_dict_values, retrieve_replacement_fields
from os import path, makedirs
from random import randint
import json


DFT_QUERIES_DELIMITER = '--^'
DFT_VERTICA_AUTH_FILE = '~/vertica_auth.json'


class VerticaError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
        
        


class VerticaEngine():
    """Vertica easy-to-use interface.
    
    Позволяет легко выполнять запросы select/execute/insert, используя
    в качестве аргумента строковую переменную, лист или путь к файлу.
        
    Parameters
    ----------
    user : str, default None
        Логин в Вертику.
    password : str, default None
        Пароль в Вертику
    auth_filepath : str filepath, default ~/vertica_auth.json
        Путь к файлу, в котором в виде json хранятся user и password
    server : str, default None
        Адрес сервера. Например 'avi-dwh27'. В случае отсутствия аргумента,
        подключение производится к рандомной ноде из диапазона 15-28.
    logger : logging.Logger object or str, default 'vertica'
        Существующий логгер, его имя, или имя для нового логгера.
        Логгер используется для записи прогресса выполнения запросов.
    query_logger : logging.Logger object or str, default 'vertica'
        Существующий логгер, его имя, или имя для нового логгера.
        Логгер используется для записи всех sql-запросов к базе.
    """

    
    def __init__(
        self,
        user=None,
        password=None,
        auth_filepath=DFT_VERTICA_AUTH_FILE,
        server=None,
        logger='vertica',
        query_logger='vertica_query_log',
        port=5433,
    ):
        
        if server is None:
            server = 'avi-dwh{}'.format(randint(15, 28))
        
        self.conn_info = {
            'user': user,
            'password': password,
            'host': server,
            'port': port,
            'database': 'DWH',
            # 10 minutes timeout on queries
            'read_timeout': 3600,
            # default throw error on invalid UTF-8 results
            #'unicode_error': 'strict',
            # SSL is disabled by default
            #'ssl': False,
            #'connection_timeout': 20
            # connection timeout is not enabled by default
        }

        if user is None:
            try:
                with open(path.expanduser(auth_filepath), 'r') as f:
                    auth = json.load(f)
            except:
                raise Exception("""You should store Vertica user/password in {}
                or provide them explicitly as kwargs""".format(auth_filepath))

            if auth.get('user', None) is None or auth.get('password', None) is None:
                raise Exception('user or password is missing')

            self.conn_info.update(auth)

        self.logger = define_logger(logger=logger, level=logging.INFO)
        self.query_logger = define_logger(
            logger=query_logger,
            level=logging.INFO,
            log_format='--[%(asctime)s]\n%(message)s;\n\n',
            to_stdout=False
        )
        
        self.reconnect()
        
    
    def reconnect(self):
        """Переподключиться. Если соединение упало или чтобы создать новую сессию.
        
        Returns
        -------
        self
        """
        try:
            self.con.close()
        except Exception:
            pass

        self.con = vertica_python.connect(**self.conn_info)
        
        return self

    @staticmethod
    def __log_msg(query):
        return '{m:<49}'.format(m=query.strip().split('\n')[0][:46] + '...')
    
    def select(self,
        queries,
        params=None,
        delimiter=DFT_QUERIES_DELIMITER,
        resume=False,
        start_query_ind=None,
        end_query_ind=None
    ):
        """Выполнить запрос на select и вернуть результат в формате DataFrame.
        
        Parameters
        ----------
        Параметры те же, что и в execute.
        """
        queries = delimiter + "\n" + queries
        return self.execute(
            queries=queries,
            params=params,
            delimiter=delimiter,
            resume=resume,
            start_query_ind=start_query_ind,
            end_query_ind=end_query_ind,
            select=True
        )
        
        
    def __execute(self, query):
        query = textwrap.dedent(query.replace(DFT_QUERIES_DELIMITER, '')).strip('\n\t ;')
        self.query_logger.info(query)
        
        with log_time(name=self.__log_msg(query), logger=self.logger):
            try:
                cur = self.con.cursor()
                cur.execute(query)
                cur.execute("commit;")
                #self.con.commit()
            except Exception:
                self.logger.exception(query)
                raise Exception(query)
   

    def execute(
        self,
        queries,
        params=None,
        delimiter=DFT_QUERIES_DELIMITER,
        resume=False,
        start_query_ind=None,
        end_query_ind=None,
        select=False
    ):
        """Выполнить запрос или очередь запросов.
        
        Parameters
        -----------
        queries : str, list or filepath
            Запрос или несколько запросов. Если запросов несколько,
            они в файле или строковым значением, то необходимо
            раздедить их разделителем, переданных в параметре delimiter.
        params : dict, default None
            Кейворды и их значение, которые нужно подставить в запрос.
        delimiter : str, default '--^'
            Разделитель запросов. Разделение дает несколько преимуществ:
            — Логирование каждой временной отсечки.
            — Возможность продолжить выполнение серии запросов, если произошла ошибка.
        resume : boolean, default False
            Продолжить выполнение серии запросов. Если в серии сбился индекс (last_query_ind),
            то можно передать правильный индекс параметром start_query_ind.
        start_query_ind : int, default None
            Стартовый индекс запроса в очереди.
        end_query_ind :  int, default None
            Конечный индекс.
        select : boolean, default False
            Вернуть результаты последнего запроса в очереди в виде DataFrame.
            Этот запрос можно написать просто в виде имени таблицы/вью.
            Тогда к нему автоматически припишется 'select * from'
        
        Returns
        -------
        DataFrame если select == True и None в противном случае
        """
        
        queries_list = parse_queries(queries, params, delimiter)
        
        if resume and start_query_ind is None:
            if not hasattr(self, 'last_query_ind'):
                self.logger.warning("Trying to resume queue with no last_query_ind defined!")
            else:
                start_query_ind = self.last_query_ind 
        elif start_query_ind is None:
            start_query_ind = 0
        
        if end_query_ind is None:
            end_query_ind = len(queries_list)
                
        for i, q in enumerate(queries_list[start_query_ind:end_query_ind]):
            self.last_query_ind = i + start_query_ind
            self.last_query = q
            if i + start_query_ind + 1 == end_query_ind and select:
                with log_time(name=self.__log_msg(q), logger=self.logger):
                    if q[:4].lower() not in ['sele', 'with']:
                        try_queries = [q, 'select * from {}'.format(q)]
                    else:
                        try_queries = [q]
                    for i, q in enumerate(try_queries):
                        try:
                            self.last_query = q
                            df = pd.read_sql(q, self.con)
                            self.last_query_ind += 1
                            return df
                        except Exception:
                            if i + 1 == len(try_queries):
                                raise Exception(q)
                            #self.logger.exception(q)
            else:
                self.__execute(q)
                self.last_query_ind += 1
            
            
    def insert(self, df, tablename, columns=None):
        """Выполнить загрузку DataFrame в таблицу.
        
        Parameters
        -----------
        df : pandas.DataFrame
            Данные для загрузки.
        tablename : str
            Имя таблицы.
        columns : list, default None.
            Столбцы для загрузки. Если не переданы, то используются все столбцы
            указанной таблицы. Имена столбцов в df и таблице должны совпадать.
        """
        
        if columns is None:
            columns = self.get_columns_list(
                tablename,
                return_as_str=False,
                method='direct'
        )
        csv = df.to_csv(
            sep='^',
            index=False,
            header=False,
            float_format='%.16g',
            columns=columns
        )
        columns_sql = ', '.join(columns)
        query = "COPY {t} ({c}) from stdin DELIMITER '^' ABORT ON ERROR DIRECT;" \
            .format(t=tablename, c=columns_sql)
        self.query_logger.info(query)
        cur = self.con.cursor()
        
        with log_time(name=self.__log_msg(query), logger=self.logger):
            cur.copy(query, csv)
    
    
    def get_columns_list(
        self,
        tablename,
        return_as_str=False,
        prefix='',
        delimiter=',\n',
        method='direct'
    ):
        """Вернуть список колонок таблицы.
        
        Paramteters
        -----------
        tablename : str
            Имя таблицы.
        return_as_str : booleatn, default False.
            Вернуть названия столбцов единой строкой. К каждому названию
            подставляется prefix, склеиваются с помощью delimiter.
        prefix : str, default ''
        delimiter : str, default ',\\n'
        method : 'v_catalog' or 'direct', default 'direct'
            Метод получения. 'v_catalog' лезет в v_catalog.columns (долго).
            Лучше использовать direct.
        
        Returns
        -------
        list, str if return_as_str
        """
        
        if method == 'v_catalog':
            table_dict = parse_tablename(tablename)
            query = """
                    select	column_name
                    from	v_catalog.columns
                    where   lower(table_schema) = lower('{schema}')
                        and lower(table_name) = lower('{table}')
                    order by column_id
                    """.format(**table_dict)
            columns = self.select(query)['column_name'].tolist()
        elif method == 'direct':
            query = """
                    select * from {} where   false
                    """.format(tablename)
            columns = self.select(query).columns.tolist()
        else:
            raise Exception("Incorrect method")
            
        if return_as_str:
            prefixed = [prefix + c for c in columns]
            return delimiter.join(prefixed)
        else:
            return columns
            
            
class VerticaIterWorker(VerticaEngine):
    """Класс-наследник VerticaEngine для удобного итеративного расчета таблиц.
        
    Parameters
    ----------
    queries : str, list or filepath
        Запрос или несколько запросов. В случае нескольких запросов необходимо
        раздедить их разделителем '--^'.
    params : dict, default None
        Кейворды и их значения, которые нужно подставлять в запрос.
        Для каждого кейворда можно передать list значений, тогда будут выполняться
        итерации со всеми возможными комбинациями параметров.
        Если используется tables_to_swap, то передавать что-либо, кроме ds,
        списком не имеет смысла.
    tables_to_swap : list or str
        Имена таблиц, которые будут обновляться инкрементально по дням.
        Эти таблицы должны быть партицированы по формуле:
        partition by year(date)*10000 + month(date)*100 + day(date)
        А даты должны быть переданы в переменной params кейвордом 'ds'
    **kwargs: VericaEngine kwargs
    """
    
    def __init__(self, queries, params, tables_to_swap=None, **kwargs):
        
        super().__init__(**kwargs)
        self.queries = queries
        self.params = params.copy()
        
        self.tables_to_swap = self.__validate_tables_to_swap(tables_to_swap)
        
     
    def __validate_tables_to_swap(self, tables_to_swap):
        if not isinstance(tables_to_swap, list):
            tables_to_swap = [tables_to_swap]
        tables = []
        if tables_to_swap:
            for t in tables_to_swap:
                table_dict = parse_tablename(t)
                tables.append(table_dict.copy())
        return tables
        
        
    def __check_queries_and_params(self):
                
        code = parse_queries(queries=self.queries, split=False)
        params_keys = set(self.params.keys())
        replacement_fields = retrieve_replacement_fields(code)
        
        if not replacement_fields.issubset(params_keys):
            raise Exception(
                """
                Replacement fields in {f} must be a subset of params keys:
                — replacement_fields: {rf}
                — params keys: {pk}
                — missing: {m}
                """. \
                format(
                    f=self.queries,
                    rf=replacement_fields,
                    pk=params_keys,
                    m=replacement_fields-params_keys
                )
            )
        
        if self.tables_to_swap and self.params.get('ds', None) is None:
            raise VerticaError("For tables swapping you should provide 'ds' (meaning date) key in params")
            
        
    def _create_tmp_tables(self):
        """Создать временные таблицы из tables_to_swap."""
        
        code = """
            drop table if exists {schema}.{table}_tmp cascade;
            create table {schema}.{table}_tmp like {schema}.{table} including projections;
        """

        for t in self.tables_to_swap:
            self.execute(code.format(**t))
        
        self.logger.debug('Temporary tables created')

    def _swap_tables(self, ds=None):
        """Свопнуть временные таблицы из tables_to_swap.
        
        Parameters
        ----------
        ds : datetime.date, default None
            Дата, обозначающая партицию для свопа.
        """
        
        code = """
            select swap_partitions_between_tables(
                '{schema}.{table}_tmp',
                {start_partition},
                {end_partition},
                '{schema}.{table}'
            );
            drop table {schema}.{table}_tmp cascade;
        """
        if ds is None:
            ds = self.current_iter_params['ds']
            
        params = {'start_partition': ds.strftime("%Y%m%d"), 'end_partition': ds.strftime("%Y%m%d")}
        
        for t in self.tables_to_swap:
            params.update(t)
            rows_count = self.select("""
                select count(*) as cnt from {schema}.{table}_tmp
                """.format(**params)).iloc[0, 0]
                
            if rows_count == 0:
                self.logger.info('{schema}.{table}_tmp has 0 rows'.format(**params))
                continue
            self.execute(code.format(**params))
            self.logger.info('Swap {schema}.{table} completed: {r} rows' \
                .format(r=rows_count, **params))
    
    
    def run_iters(
        self,
        start_iter_ind=None,
        end_iter_ind=None,
        resume=False,
        try_only=False,
        start_query_ind=None,
        end_query_ind=None
    ):
        """Запустить итеративный расчет.
        
        Parameters
        ----------
        start_iter_ind : int, default None
            Стартовый индекс итерации.
        end_iter_ind : int, default None
            Конечный индекс итерации.
        resume : boolean, default False
            Продолжить выполнение. Если в текущей серии сбился индекс (last_query_ind),
            то можно передать правильный индекс параметром start_query_ind.
        try_only : boolean, default False
            Режим для отладки sql-кода. Выполняется только первая итерация.
            Рамки можно задать параметром end_query_ind.
        start_query_ind : int, default None
            Стартовый индекс в очереди запросов. Используется в случае resume.
        end_query_ind : int, default None
            Конечный индекс в очереди запросов. Используется в случае try_only.
        """

        self.__check_queries_and_params()
        self.iters = iter_dict_values(self.params)
        self.logger.info("{} iters in total".format(len(self.iters)))
        
        if start_iter_ind is None:
            start_iter_ind = 0
        
        if end_iter_ind is None:
            end_iter_ind = len(self.iters)
        
        for iter_ind, iter_params in enumerate(self.iters[start_iter_ind:end_iter_ind]):
            
            if not resume:
                self.reconnect()
                self.current_iter_ind = iter_ind + start_iter_ind
                self.current_iter_params = iter_params
                if self.tables_to_swap:
                    self._create_tmp_tables()               
            
            if not try_only:
                end_query_ind = None
            
            if resume and start_query_ind is None:
                try:
                    start_query_ind = self.last_query_ind_in_iters
                except Exception:
                    pass
            
            try:
                self.execute(
                    queries=self.queries,
                    resume=resume,
                    params=self.current_iter_params,
                    start_query_ind=start_query_ind,
                    end_query_ind=end_query_ind
                )
            except Exception:
                self.last_query_ind_in_iters = self.last_query_ind
                raise

            if self.tables_to_swap and not try_only:
                self._swap_tables()
            
            self.logger.info('iter {} completed'.format(self.current_iter_params))
            
            if try_only:
                break
            
            resume = False
        
    
def parse_queries(queries, params=None, delimiter=DFT_QUERIES_DELIMITER, split=True):
    """Распарсить строку или файл с запросами и вернуть очередь запросов в виде листа.
    
    Parameters
    ----------
    queries : str, list or filepath
        Запрос или несколько запросов. В случае нескольких запросов необходимо
        раздедить их разделителем, переданным в параметре delimiter.
    params : dict, default None
        Кейворды и их значение, которые нужно подставить в запрос.
    delimiter : str, default '--^'
        Разделитель.
    split : boolean, default True
        Если True, вернуть запросы в листе. В противном случае — строковым
        значением, склеенным через delimiter.
    """
    
    if isinstance(queries, str):
        if len(queries) < 100 and path.isfile(queries):
            with open(queries, 'r', encoding='utf-8') as f:
                queries = f.read()
        else:
            pass
    elif isinstance(queries, list):
        queries = delimiter.join(queries)
    else:
        raise Exception('Incorrect queries type')
    
    if params is None:
        pass
    elif isinstance(params, dict):
        queries = queries.format(**params)
    else:
        raise VerticaError("'params' variable should be of type 'dict'")
    
    queries_list = list(e.strip('\n\t ;') for e in queries.split(delimiter))
    queries_list = list(e for e in queries_list if len(e) > 0)
    
    if len(queries_list) >= 1:
        if split:
            return queries_list
        else:
            return delimiter.join(queries_list)
    else:
        raise VerticaError('Nothing to parse')


def parse_tablename(tablename, schema='public'):
    """Распарсить имя таблицы в дикт {schema: '...', 'table': '...'}."""
    table_dict = {}
    table_parsed = tablename.split('.')
    if len(table_parsed) == 2:
        table_dict['schema'], table_dict['table'] = table_parsed[0], table_parsed[1]
    elif len(table_parsed) == 1:
        table_dict['schema'], table_dict['table'] = schema, table_parsed[0]
    else:
        raise VerticaError('Incorrect table name' + tablename)
    return table_dict