import numpy as np
import pandas as pd
import itertools
import datetime
import logging
import sys
from .utils import define_logger, filter_df, iter_dict_values
from bootstrapped.bootstrap import bootstrap, bootstrap_ab
from bootstrapped.stats_functions import sum, mean, std
from bootstrapped.compare_functions import difference, percent_change, ratio, percent_difference
from scipy import sparse
from scipy import stats


def mann_whitney_u(test, ctrl, alpha):
    """Mann-Whitney U-test
    Вернуть significance, alternative, p-value.
    
    significance:
        -1 : если p-value <= alpha и alternative == 'less'
        1  : если p-value <= alpha и alternative == 'greater'
        0  : если p-value > alpha в обоих случаях
    """
    
    stat_values = {}
    for significance, alternative in zip([-1, 1], ['less', 'greater']):
        try:
            stat_value = stats.mannwhitneyu(test, ctrl, use_continuity=True, alternative=alternative)
            if stat_value.pvalue <= alpha:
                return {
                    'significance': significance,
                    'alternative': alternative,
                    'p_value': stat_value.pvalue
                }
            stat_values[alternative] = stat_value
        except ValueError:
            return {
                'significance': 0,
                'alternative': None,
                'p_value': None
            }
    
    if stat_values['less'].pvalue < stat_values['greater'].pvalue:
        return {
            'significance': 0,
            'alternative': 'less',
            'p_value': stat_values['less'].pvalue
        }
    else:
        return {
            'significance': 0,
            'alternative': 'greater',
            'p_value': stat_values['greater'].pvalue
        }

        
COMPARE_FUNCS = {
    'difference': difference,
    'percent_change': percent_change,
    'ratio': ratio,
    'percent_difference': percent_difference,
    'mann_whitney_u': mann_whitney_u,
}

def median(values, axis=1):
    '''Returns the median of each row of a matrix'''
    if isinstance(values, sparse.csr_matrix):
        ret = values.median(axis=axis)
        return ret.A1
    else:
        return np.median(np.asmatrix(values), axis=axis).A1

STAT_FUNCS = {
    'sum': sum,
    'mean': mean,
    'std': std,
    'median': median,
}

DFT_NUM_BOOTSTRAP_SAMPLES = 10000
DFT_ALPHA = 0.05

DFT_NUM_THREADS = 4
DFT_ITER_BATCH_SIZE = 100


from jsonschema import Draft4Validator, validators

def __extend_validator_with_default(validator_class):
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for property, subschema in properties.items():
            if "default" in subschema:
                instance.setdefault(property, subschema["default"])

        for error in validate_properties(
            validator, properties, instance, schema,
        ):
            yield error

    return validators.extend(
        validator_class, {"properties" : set_defaults},
    )

def validate_params(params, schema):
    DefaultValidatingDraft4Validator = __extend_validator_with_default(Draft4Validator)
    DefaultValidatingDraft4Validator(schema).validate(params)
    return params
    

class Bootstrapper(object):
    """
    Движок для расчета метрик и значимости по аб-тесту.
    
    Parameters
    ----------
    data : pd.DataFrame
        Данные. Обязательные колонки:
        uid         : id экспериментальной единицы (cookie, user, etc)
        ds          : дата (aka. event_date)
        aa_ab       : идентификатор периода AA, AB, AC ...
        split_group : идентификатор сплит-группы
        + столбцы с измерениями (хотя бы один)
        + дименшены (опционально)

    measures : ['measure', {property: value}]
        Столбцы с измерениями.
        measure    : название соотв. столбца
        properties :
            "balancing": { "type": "number", "minimum": 0, "maximum": 100, "default": 100 }
                         Балансировка выборки по по соотв. измерению.
                         Для примера, 99.8 возьмет 0.2% самых больших значений
                         и присвоит им медиану.
            "agg_func": { "enum": ["sum", "max", "min"], "default": "sum" }
                        Как агрегировать измерение.        
        
    dimensions : ['dimension', {property: value}]
        Столбцы с дименшенами, в разрезе которых происходит расчет. 
        dimension  : название соотв. столбца
        properties :
            "mask"    : { "enum": ["AV", "A", "V"], "default": "V" }
                        "A" означает Any, "V" — Value.
                        Итерации генерируются по всевозможным комбинациями.
            "include" : { "type": "array" }
                        list значений дименшена, чтобы только их включить в расчет.
            "exclude" : { "type": "array" }
                        list значений дименшена для исключения. Не играет роли,
                        если определен "include"
        
    metrics : ['metric', {property: value}]
        Список метрик и метод их расчета.
        metric  : название метрики
        properties :
            "numenator"   : { "enum": list(self.measures.keys()), default 'metric' }
                            Измерение в числителе.
            "denominator" : { "enum": list(self.measures.keys()) + [None],
                              "default": None }
                            Измерение в знаменателе (опционально).
            "stat_func"   : { "enum": ["sum", "mean", "std", "median"],
                              "default": "sum" }
                            Статистика.
            "bootstrap"   : { "type": "integer", "minimum": 1, "maximum": 10000,
                              "default": 10000 },
                            Количество сэмплов при бутстрапе.
            "is_pivotal"  : { "type" : "boolean", "default": False },
                            Значение метрики считать по всей выборке.

            "alpha"       : { "type": "number", "minimum": 0, "maximum": 1,
                              "default": 0.05 }
                            Уровень значимости.
    
    dss : list of dates, default None
        Фильтр на дни. Если None, то используются все доступные дни.
        
    split_groups_to_compare_with : 
        Список групп для расчета разницы. Обычно нужно передать только
        контрольную группу.

    control_groups : list
        Список контрольных групп. Используется для правильной сортировки при
        расчете разницы между группами. Чтобы аплифт в отчете был положительным,
        а даунлифт — отрицательным.
    
    ds_windows : list of integers
        Список размеров скользящих окон для расчета метрик. По дефолту только 1,
        т. е. индивидуальный расчет для каждого дня.
        
    existing_estimates : DataFrame
        Уже рассчитанные ранее результаты, чтобы считать только новое.
        
    logger : Logger object or string
        Существующий логгер (или его название) или название для нового.
    """
    
    def __init__(
        self,
        data,
        measures,
        metrics,
        dimensions,
        dss=None,
        split_groups_to_compare_with=None,
        control_groups=None,
        compare_funcs=None,
        ds_windows=None,
        existing_estimates=None,
        logger='ab_calculator'
    ):
        self.logger = define_logger(logger=logger, level=logging.INFO)
        self.__validate_data(data)
        
        self.__validate_measures(measures)
        self.__validate_dimensions(dimensions)
        self.__validate_metrics(metrics)
        
        self.__validate_dss(dss)
        self.__validate_ds_windows(ds_windows)

        self.__validate_control_groups(control_groups)
        self.__validate_split_groups_to_compare_with(split_groups_to_compare_with)
        self.__validate_compare_funcs(compare_funcs)
        
        self.__define_iters()
        
        self.__validate_existing_estimates(existing_estimates)
    
    def __validate_data(self, data):
    
        data.split_group = data.split_group.astype(str)
        self.data_columns = set(data.columns.tolist())
        self.data_required_columns = {'uid', 'ds', 'aa_ab', 'split_group'}
        self.data_other_columns = self.data_columns - self.data_required_columns
        self.split_groups = sorted(data.split_group.unique().tolist())
        self.dss = sorted(data.ds.unique().tolist())
        data_missing_columns = self.data_required_columns - self.data_columns
        if data_missing_columns:
            raise Exception("missing columns in data: {}".format(data_missing_columns))
        if not self.data_other_columns:
            raise Exception("data has no any measure columns")
        self.data = data
    
    def __validate_list_params(self, arg, schema, to_string=False):
        if arg is None:
            arg = []
        elif not isinstance(arg, list):
            arg = [arg]
        
        if to_string:
            arg = [str(a) for a in arg]
        
        valid_arg = validate_params(arg, schema)
        return valid_arg
        
    
    def __validate_task_params(self, args, schema):
        if isinstance(args, str):
            args = [args]
        
        valid_args = {}
        for arg in args:
            if isinstance(arg, str):
                arg_name = arg
                params = {}
            elif    isinstance(arg, list) \
                and len(arg) in [1, 2] \
                and isinstance(arg[0], str) \
                and (len(arg) == 2 and isinstance(arg[1], dict) or len(arg) == 1):
            
                arg_name = arg[0]
                if len(arg) == 1:
                    params = {}
                else:
                    params = arg[1]
                    
            else:
                raise Exception('Incorrect parameter format: {}'.format(arg))
            
            valid_args[arg_name] = validate_params(params, schema)
            
        return valid_args
    
    
    @staticmethod
    def __extract_param_dict(task_params, param_name):
        return {k: v.get(param_name, None) for k, v in task_params.items()}
    
    
    def __validate_measures(self, measures):
    
        self.MEASURE_PARAMETERS_SCHEMA = {
            "type": "object",
            "properties": {
                "balancing": { "type": "number", "minimum": 0, "maximum": 100, "default": 100 },
                "agg_func": { "enum": ["sum", "max", "min"], "default": "sum" },
            }
        }
        
        measures = self.__validate_task_params(
            measures,
            self.MEASURE_PARAMETERS_SCHEMA
        )
        
        self.measures = {k: v for k, v in measures.items() \
            if k in self.data_other_columns}
        
        
    def __validate_dimensions(self, dimensions):
        
        self.DIMENSION_PARAMETERS_SCHEMA = {
            "type": "object",
            "properties": {
                "mask": { "enum": ["AV", "A", "V"], "default": "V" },
                "include": { "type": "array" },
                "exclude": { "type": "array" }
            }
        }
        
        dimensions_raw = self.__validate_task_params(
            dimensions,
            self.DIMENSION_PARAMETERS_SCHEMA
        )
        
        dimensions = {
            k: v for k, v in dimensions_raw.items() \
            if k in self.data_other_columns \
            and v['mask'] != 'A'
            and k not in self.measures}
        
        dimensions.update({'aa_ab': {'mask': 'V'}})
        
        for d in dimensions:
            include = dimensions[d].get('include', [])
            exclude = dimensions[d].get('exclude', [])
            
            dimension_values = self.data[d].unique().tolist()
            
            constraints = None
            if include:
                constraints = [v for v in dimension_values if v in include]
            elif exclude:
                constraints = [v for v in dimension_values if v not in exclude]
            
            if constraints:
                dimensions[d]['constraints'] = constraints
        
        self.dimensions = dimensions
        
                
    def __validate_metrics(self, metrics):
        
        self.METRIC_PARAMETERS_SCHEMA = {
            "type": "object",
            "properties": {
                "numenator": { "enum": list(self.measures.keys()) },
                "denominator": { "enum": list(self.measures.keys()) + [None], "default": None },
                "stat_func": { "enum": ["sum", "mean", "std", "median"], "default": "sum" },
                "bootstrap": { "type": "integer", "minimum": 1, "maximum": 10000, "default": DFT_NUM_BOOTSTRAP_SAMPLES },
                "is_pivotal": { "type" : "boolean", "default": False },
                "alpha": { "type": "number", "minimum": 0, "maximum": 1, "default": DFT_ALPHA }
            }
        }
        
        metrics = self.__validate_task_params(
            metrics,
            self.METRIC_PARAMETERS_SCHEMA
        )
        
        for k, v in metrics.items():
            if not v.get('numenator', None):
                metrics[k]['numenator'] = k
        
        self.metrics = metrics
        
    
    def __validate_dss(self, dss):

        if dss is None:
            dss = self.dss

        self.DSS_SCHEMA = {
            "type": "array",
            "items": { "enum": self.dss }
        }
        
        self.dss = self.__validate_list_params(dss, self.DSS_SCHEMA)        
        
        
    def __validate_ds_windows(self, ds_windows):

        if ds_windows is None:
            ds_windows = [1]

        self.DS_WINDOWS_SCHEMA = {
            "type": "array",
            "items": { "enum": list(range(1, 36)) },
        }
        
        self.ds_windows = self.__validate_list_params(ds_windows, self.DS_WINDOWS_SCHEMA)
    

    def __validate_control_groups(self, control_groups):
        
        self.SPLIT_GROUPS_SCHEMA = {
            "type": "array",
            "items": { "enum": self.split_groups }
        }
    
        self.control_groups = self.__validate_list_params(
            control_groups,
            self.SPLIT_GROUPS_SCHEMA,
            to_string=True
        )
        
    
    def __validate_split_groups_to_compare_with(self, split_groups_to_compare_with):

        split_groups_to_compare_with = self.__validate_list_params(
            split_groups_to_compare_with,
            self.SPLIT_GROUPS_SCHEMA,
            to_string=True
        )
        
        def sortkey(x):
            if x in self.control_groups:
                return 1
            elif x in split_groups_to_compare_with:
                return 2
            else:
                return 3
        
        all_combinations = list(sorted(i, key=sortkey) for i in itertools.combinations(self.split_groups, 2))
        
        self.split_groups_to_compare = [
            i for i in all_combinations \
            if i[0] in split_groups_to_compare_with \
            or i[1] in split_groups_to_compare_with
        ]
    
    
    def __validate_compare_funcs(self, compare_funcs):
        
        self.COMPARE_FUNCS_SCHEMA = {
            "type": "array",
            "items": { "enum": list(COMPARE_FUNCS.keys()) },
            "default": ['percent_change', 'mann_whitney_u']
        }
        
        self.compare_funcs = self.__validate_list_params(
            compare_funcs,
            self.COMPARE_FUNCS_SCHEMA
        )
    
    
    def __define_iters(self):
        iter_columns = list(self.dimensions.keys()) + ['ds']
        
        ordinary_iters = self.data[self.data.ds.isin(self.dss)] \
            .groupby(iter_columns).sum() \
            .reset_index()[iter_columns]
        
        extra_iters = iter_dict_values({'ds_window': self.ds_windows})
        
        iters = pd.DataFrame()
        
        dimensions_masks = self.__extract_param_dict(self.dimensions, 'mask')
        dimensions_constraints = self.__extract_param_dict(self.dimensions, 'constraints')
        
        iter_masks = iter_dict_values(dimensions_masks, process_strings_as_list=True)
        
        
        for imask, iother in itertools.product(iter_masks, extra_iters):
            groupby = list(k for k, v in imask.items() if v == 'V') + ['ds']
            itr = ordinary_iters.groupby(groupby).size().reset_index()[groupby]
            for k, v in dimensions_constraints.items():
                if k not in groupby or not v: continue
                itr = itr[itr[k].isin(v)].copy()
            for k, v in iother.items():
                itr[k] = v
            iters = iters.append(itr, ignore_index=True)
        
        self.iters = iters.sort_values(iters.columns[::-1].tolist(), na_position='first').reset_index(drop=True)
        

    def __validate_existing_estimates(self, existing_estimates):
        estimates_columns = self.iters.columns.tolist() \
                + ['split_group', 'metric', 'compare_func'] \
                #+ ['value', 'value_lower', 'value_upper']
        
        if existing_estimates is None:
            existing_estimates = pd.DataFrame([], columns=estimates_columns)
        elif not isinstance(existing_estimates, pd.DataFrame):
            Exception("Incorrect existing_estimates format")
        
        estimates = pd.DataFrame([], columns=estimates_columns)
        
        self.estimates = estimates
        self.existing_estimates = existing_estimates[estimates_columns]
    

    def _get_iter_df(self, iter_split_group):
        
        iter_filter = iter_split_group.dropna()
        
        dims_to_filter = [c for c in iter_filter.index if c not in ['ds', 'ds_window']]
        
        dim_ftr = (self.data[dims_to_filter] == iter_filter[dims_to_filter]).all(axis=1)
        date_ftr = (self.data.ds > (iter_filter.ds - datetime.timedelta(days=iter_filter.ds_window))) & \
                   (self.data.ds <= iter_filter.ds)
        df0 = self.data[dim_ftr & date_ftr]
        actual_ds_window = df0.ds.unique().shape[0]
        if (df0.ds == iter_filter.ds).sum() == 0:
            return None, None
        
        measures_agg_funcs = self.__extract_param_dict(self.measures, 'agg_func')
        return df0.groupby('uid').aggregate(measures_agg_funcs), actual_ds_window

        
    @staticmethod
    def _handle_extremes(arr, percentile=99.8):
        if percentile >= 100:
            return arr
        bound = np.percentile(arr, q=percentile)
        new_value = arr.median()
        arr[arr > bound] = new_value
        return arr
        
    
    def _is_sub_iter_exists(self, sub_iter, only_estimates=False):
        
        if only_estimates:
            all_estimates = self.estimates
        else:
            all_estimates = pd.concat([self.estimates, self.existing_estimates])
        
        if filter_df(all_estimates.fillna(-1), sub_iter.fillna(-1)).shape[0] > 0:
            #self.logger.debug('sub_iter {d} is already in estimates'.format(d=sub_iter.to_dict()))
            return True
        
        return False
        
    def run_iters(self, num_threads=DFT_NUM_THREADS, iteration_batch_size=DFT_ITER_BATCH_SIZE):
        """
        Запустить расчет итераций.
        
        Parameters
        ----------
        num_threads : integer, default 4
            Количество потоков
        iteration_batch_size : integer, default 100
            Размер батча при бутстрап-сэмплинге чтобы не забивать память.
            Значения от 1 до bootstrap. По достижении порога батча,
            считаются статистики и очищается занимаемая память.
        """
       
        self.logger.info('{} iters to go'.format(self.iters.shape[0]))
                
        for c, iter in self.iters.iterrows():
            iter_data = {}
            iter_estimates = pd.DataFrame()
            for sg in self.split_groups:
                iter_data[sg] = {}
                sub_iter = iter.copy()
                sub_iter['split_group'] = sg
                df, actual_ds_window = self._get_iter_df(sub_iter)
                if df is None:
                    self.logger.debug('iter {d} is empty'.format(d=sub_iter.to_dict()))
                    continue
                
                sub_iter.ds_window = actual_ds_window
                skip = True
                
                for metric in self.metrics:
                    sub_iter['metric'] = metric
                    if not self._is_sub_iter_exists(sub_iter, only_estimates=True):
                        skip = False
                        break
                    
                if skip: continue

                for measure, params in self.measures.items():
                    if params['balancing'] < 100:
                        df[measure] = self._handle_extremes(
                            df[measure],
                            percentile=params['balancing']
                        )
                
                for metric in self.metrics:
                    sub_iter['metric'] = metric
                    numenator = df[self.metrics[metric]['numenator']].values
                    if self.metrics[metric]['denominator'] is not None:
                        denominator = df[self.metrics[metric]['denominator']].values
                    else:
                        denominator = None
                    iter_data[sg][metric] = {'numenator': numenator, 'denominator': denominator}
                    
                    if self._is_sub_iter_exists(sub_iter): continue
                    
                    stat_func = STAT_FUNCS[self.metrics[metric]['stat_func']]
                    
                    bs_estimate =  bootstrap(
                        values=numenator,
                        stat_func=stat_func,
                        denominator_values=denominator,
                        num_iterations=self.metrics[metric]['bootstrap'],
                        alpha=self.metrics[metric]['alpha'],
                        is_pivotal=self.metrics[metric]['is_pivotal'],
                        num_threads=num_threads,
                        iteration_batch_size=iteration_batch_size,
                        return_distribution=False,
                    )
                    
                    is_fake_bs = self.metrics[metric]['bootstrap'] == 1
                    
                    metric_params = self.metrics[metric].copy()
                    
                    iter_row = sub_iter.copy()
                
                    iter_row['value'] = bs_estimate.value
                    iter_row['value_lower'] = bs_estimate.lower_bound if not is_fake_bs else None
                    iter_row['value_upper'] = bs_estimate.upper_bound if not is_fake_bs else None
                    
                    iter_row = iter_row.append(pd.Series(metric_params))
                    
                    iter_estimates = iter_estimates.append(iter_row, ignore_index=True)
            
            
            for sgs_pair, metric, compare_func in itertools.product(
                self.split_groups_to_compare,
                self.metrics.keys(),
                self.compare_funcs
            ):
                if skip: break
                
                test = iter_data[sgs_pair[1]][metric]['numenator']
                ctrl = iter_data[sgs_pair[0]][metric]['numenator']
                test_denominator = iter_data[sgs_pair[1]][metric]['denominator']
                ctrl_denominator = iter_data[sgs_pair[0]][metric]['denominator']
                alpha = self.metrics[metric]['alpha']
                
                    
                sgs_pair_str = '|'.join(sgs_pair)
                if compare_func == 'difference':
                    sgs_pair_str += ' (difference)'
                elif compare_func == 'percent_change':
                    sgs_pair_str += ' (% change)'
                elif compare_func == 'ratio':
                    sgs_pair_str += ' (ratio)'
                elif compare_func == 'percent_difference':
                    sgs_pair_str += ' (% difference)'
                elif compare_func == 'mann_whitney_u':
                    sgs_pair_str += ' (MW. U-test)'
                    
                sub_iter.split_group = sgs_pair_str
                sub_iter.metric = metric
                
                
                if     iter_data.get(sgs_pair[0], None) is None \
                    or iter_data.get(sgs_pair[1], None) is None:
                    self.logger.debug('iter {d} is empty'.format(d=sub_iter.to_dict()))
                    continue
                    
                if self._is_sub_iter_exists(sub_iter):
                    continue
                
                stat_func = STAT_FUNCS[self.metrics[metric]['stat_func']]
                
                
                if compare_func == 'mann_whitney_u':
                     
                    if test_denominator is not None:
                        test_mw = test / test_denominator
                        ctrl_mw = ctrl / ctrl_denominator
                    else:
                        test_mw = test
                        ctrl_mw = ctrl
                        
                    mw_estimate = mann_whitney_u(test_mw, ctrl_mw, alpha)
                    
                    bs_estimate = bootstrap_ab(
                        test=test,
                        ctrl=ctrl,
                        test_denominator=test_denominator,
                        ctrl_denominator=ctrl_denominator,
                        stat_func=stat_func,
                        compare_func=COMPARE_FUNCS['percent_change'],
                        num_iterations=1,
                        is_pivotal=True,
                        alpha=alpha,
                        scale_test_by=1.0,
                        iteration_batch_size=1,
                        num_threads=1,
                        return_distribution=False,
                    )
                    
                    metric_params = self.metrics[metric].copy()
                    metric_params['compare_func'] = compare_func
                    metric_params.pop('bootstrap')
                    metric_params.pop('is_pivotal')
                
                    iter_row = sub_iter.copy()
            
                    iter_row['value'] = bs_estimate.value
                    
                    iter_row = iter_row.append(pd.Series(metric_params))
                    iter_row = iter_row.append(pd.Series(mw_estimate))

                    iter_estimates = iter_estimates.append(iter_row, ignore_index=True)
            
                else:
                    
                    bs_estimate = bootstrap_ab(
                        test=test,
                        ctrl=ctrl,
                        test_denominator=test_denominator,
                        ctrl_denominator=ctrl_denominator,
                        stat_func=stat_func,
                        compare_func=COMPARE_FUNCS[compare_func],
                        num_iterations=self.metrics[metric]['bootstrap'],
                        is_pivotal=self.metrics[metric]['is_pivotal'],
                        alpha=alpha,
                        scale_test_by=1.0,
                        iteration_batch_size=iteration_batch_size,
                        num_threads=num_threads,
                        return_distribution=False,
                    )

                    is_fake_bs = self.metrics[metric]['bootstrap'] == 1
                    
                    metric_params = self.metrics[metric].copy()
                    metric_params['compare_func'] = compare_func
                    
                    iter_row = sub_iter.copy()
                    
                    iter_row['value'] = bs_estimate.value
                    iter_row['value_lower'] = bs_estimate.lower_bound if not is_fake_bs else None
                    iter_row['value_upper'] = bs_estimate.upper_bound if not is_fake_bs else None
                    iter_row['significance'] = bs_estimate.get_result() if not is_fake_bs else None
                    
                    iter_row = iter_row.append(pd.Series(metric_params))

                    iter_estimates = iter_estimates.append(iter_row, ignore_index=True)
            
            self.estimates = self.estimates.append(iter_estimates, ignore_index=True)
            self.logger.info('iter {c} {d} completed'.format(c=c+1, d=iter.to_dict()))
        
        self.logger.info('all done!')  
        return self
        
                
    def export_to_vertica(self, con_vertica, tablename='public.buyerx_ab', **details):
        """
        Результаты могут быть экспортированы в таблицу со следующей структурой:
            ds date,
            metric varchar(64),
            split_group varchar(256),
            aa_ab varchar(3),
            ds_window varchar(64),
            value float,
            value_lower float,
            value_upper float,
            alpha float,
            bootstrap int,
            is_pivotal boolean,
            numenator varchar(256),
            denominator varchar(256),
            stat_func varchar(64),
            insert_datetime timestamp(0),
            +
            dimensions
            
        Parameters
        ----------
        con_vertica : VerticaEngine object
        **details : kwargs
            Доп. инфо для вставки в таблицу. Обычно используются:
                — ab_label (must have) : метка теста, чтобы фильтровать в отчете.
                — jira_issue : номер соответствующего таска.        
        """
        
        target_columns = con_vertica.get_columns_list(tablename)
        
        data_to_export = self.estimates.copy()
        
        data_to_export['insert_datetime'] = datetime.datetime.now()
        
        if details:
            for k, v in details.items():
                data_to_export[k] = v
        
        for c in target_columns:
            if c not in data_to_export:
                data_to_export[c] = None
        
        con_vertica.insert(data_to_export, tablename, target_columns)
        
        return self
    