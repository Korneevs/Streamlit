import datetime
import json
from pathlib import Path
import clickhouse_connect
from string import Formatter
from copy import deepcopy
import hashlib
import typing as tp


def read_json_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = f.read()
        data = json.loads(data)
        return data
    

def make_clean_sql_query(raw_sql: str):
    sql_array = raw_sql.split('\n')
    sql_query = "\n".join([part for part in sql_array if not part.isspace() and not part == ''])
    return sql_query


def fill_query(not_filled_query: str, full_keys_dict: tp.Dict[str, str]) -> str:
        """
            Заполнить not_filled_query необходимыми ключами из full_keys_dict
        """

        fieldnames = set(fname for _, fname, _, _ in Formatter().parse(not_filled_query) if fname)
        
        necessary_dict_information = {key: full_keys_dict[key] for key in fieldnames}
        return not_filled_query.format(**necessary_dict_information)


def fill_query_final_added_dict(not_filled_query: str, full_keys_dict: tp.Dict[str, str]) -> str:
    """
        Заполнить not_filled_query необходимыми ключами из full_keys_dict
    """

    fieldnames = set(fname for _, fname, _, _ in Formatter().parse(not_filled_query) if fname)

    necessary_dict_information = {key: full_keys_dict[key] for key in fieldnames}
    return necessary_dict_information


def clean_array_from_strings_if_possible(array):
    result = []
    for value in array:
        try:
            value = int(value)
        except (TypeError, ValueError):
            pass
        result.append(value)
    return result

def read_sql_file(file_path: str) -> str:
    """
        Читает sql текст из файла, лежащего по пути file_path
    """
    with open(file_path, 'r') as sql_file:
        sql_query = list(sql_file.readlines())

    sql_query = make_clean_sql_query("".join(sql_query))
    return sql_query