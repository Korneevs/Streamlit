import calendar
import pandas as pd
import logging
from os import path, makedirs
from log_helper import configure_logger, DEFAULT_FMT
from itertools import product
import datetime
import string

DFT_LOG_DIR = path.expanduser('~/log/')

def define_logger(logger, log_dir=DFT_LOG_DIR, log_format=DEFAULT_FMT, **kwargs):
    """Вернуть логгер. Если на вход подан логгер, то его же и возвращаем.
    Если строка, то делаем логгер с таким именем."""
    if isinstance(logger, logging.Logger):
        return logger
    else:
        if isinstance(logger, str):
            logger_name = logger
            filename = logger_name + '.log'
        else:
            raise Exception('Incorrect logger type')
        
        log_dir = path.expanduser(log_dir)
        makedirs(log_dir, exist_ok=True)
        
        return configure_logger(
            logger_name=logger_name,
            log_dir=log_dir,
            log_format=log_format,
            filename=filename,
            **kwargs
        )

        
def current_date_range(start_date, end_date=None, ascending=True):
    """Вернуть лист дней, начиная со start_date по вчерашний день.
    Можно указать end_date и сортировку."""
    if end_date is None:
        end_date = datetime.date.today() - datetime.timedelta(days=1)
    return [d.date() for d in pd.date_range(start_date, end_date).sort_values(ascending=ascending)]
        
        
def filter_df_by_df(df_to_filter, filter_df, exclude=False):
    """Отфильтровать df с помощью другого df."""
    keys = filter_df.columns.tolist()
    ind_to_ftr = df_to_filter.set_index(keys).index
    ftr_ind = filter_df.set_index(keys).index
    ftr = ind_to_ftr.isin(ftr_ind)
    if exclude:
        ftr = ~ftr
    return df_to_filter[ftr]
    
    
def filter_df_by_dict(df_to_filter, filter_dict, exclude=False):
    """Отфильтровать df с помощью дикта."""
    filter_df = pd.DataFrame(filter_dict, index=[0])
    return  filter_df_by_df(df_to_filter, filter_df, exclude)

    
def filter_df_by_series(df_to_filter, filter_series, exclude=False):
    """Отфильтровать df с помощью Series."""
    ftr = (df_to_filter[filter_series.index] == filter_series).all(axis=1)
    if exclude:
        ftr = ~ftr
    return  df_to_filter[ftr]

    
def filter_df(df_to_filter, ftr, exclude=False):
    """Отфильтровать df с помощью другого df, дикта, или Series."""
    if isinstance(ftr, pd.DataFrame):
        return filter_df_by_df(df_to_filter, ftr, exclude)
    elif isinstance(ftr, dict):
        return filter_df_by_dict(df_to_filter, ftr, exclude)
    elif isinstance(ftr, pd.Series):
        return filter_df_by_series(df_to_filter, ftr, exclude)
    else:
        raise Excepition('ftr should be dict, Dataframe or Series')

        
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    l = list(l)
    for i in range(0, len(l), n):
        yield l[i:i + n]
                                          

def add_months(sourcedate, months):
    """Добавить месяцев к дате."""
    month = sourcedate.month - 1 + months
    year = int(sourcedate.year + month / 12)
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def dragonball(tables, gap_step=' '):
    """Вернуть сгенерированных sql драгонболла для списка таблиц.
    
    На вход переменной tables нужно подать лист таблиц в подрядке, в котором планируется джойнить.
    Формат строки: ('table_name', 'table_alias', 'join_table_alias', 'table_key', 'join_table_key', 'join_type'),
    Формула для генерации в Excel: ="('"&A1&"', '"&B1&"', '"&C1&"', '"&D1&"', '"&E1&"', '"&F1&"'),"
    """
    n_tables = len(tables)
    output = ''
    for i in range(n_tables):
        gap =  gap_step * i
        tablename = tables[n_tables-i-1][0]
        alias = tables[n_tables-i-1][1]
    
        if i != n_tables-1:
            join = '{t} join /*+jtype(fm)*/'.format(t=tables[n_tables-1-i][5])
        else:
            join = ''
        s = '{g}{t} {a} {j}\n'.format(g=gap, j=join, t=tablename, a=alias)
        output += s

    for i in range(1, n_tables):
        gap =  gap_step * (n_tables-i-1)
        column1 = tables[i][3]
        column2 = tables[i][4]
        alias1 = tables[i][1]
        if tables[i][2] != '':
            alias2 = tables[i][2]
        else:
            alias2 = tables[i-1][1]

        s = '{g}on {a1}.{c1} = {a2}.{c2}\n'.format(g=gap, c1=column1, c2=column2, a1=alias1, a2=alias2)
        output += s
    
    return output


def iter_dict_values(dct, process_strings_as_list=False):
    """Преобразовать дикт листов в лист диктов, где всевозможные комбинации
    получены декартовым произведением значений из листов."""
    for k in dct:
        is_iterable = hasattr(dct[k], '__iter__')
        is_string = isinstance(dct[k], str)
        if (is_iterable and not is_string) \
            or (is_string and process_strings_as_list):
            continue
        else:
            dct[k] = [dct[k]]
    return list(dict(zip(dct, x)) for x in product(*dct.values()))
    

def retrieve_replacement_fields(format_string):
    """Вернуть имена кейвордов из строки."""
    return set([t[1] for t in string.Formatter().parse(format_string) if t[1] is not None])