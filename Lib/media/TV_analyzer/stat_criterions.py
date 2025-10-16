import numpy as np
import scipy.stats as sps
from collections import namedtuple
from joblib import Parallel, delayed


# Структура для хранения результатов пост-нормировки.
PostNormedComparisonResults = namedtuple('PostNormedComparisonResults', 
                                         ['pvalue', 'effect', 'ci_length', 'left_bound', 'right_bound'])


def get_checking_alphas(alphas):
    """
    Преобразует массив альфа-значений в массив значений 
    для проверки доверительных интервалов.

    Параметры:
        alphas (array): Массив альфа-значений.

    Возвращает:
        array: Отсортированный массив граничных значений.
    """
    
    alphas = np.array(alphas)
    check_alphas = np.sort(np.concatenate([(1 - alphas) / 2, (1 + alphas) / 2]))
    
    return check_alphas


def bootstrap_main_statistics(sum_control_test_dict):
    """
    Основная функция расчёта отношения эффекта тестовой группы к контрольной.

    Параметры:
        sum_control_test_dict (dict): Словарь с суммами по контрольной и тестовой группе.

    Возвращает:
        float: Относительный эффект теста.
    """
    
    test_relation = sum_control_test_dict['test_after_sum'] / (sum_control_test_dict['test_before_sum'] + 1e-5)
    control_relation = sum_control_test_dict['control_after_sum'] / (sum_control_test_dict['control_before_sum'] + 1e-5)
    
    return test_relation / (control_relation + 1e-5) - 1


def _bootstrap_body(num, alpha, batch_size, n_jobs, bootstrap_function, helping_function, arrays_dict, additional_information_dict):
    """
    Выполняет бутстрап для оценки эффекта с учётом распараллеливания.

    Параметры:
        num (int): Количество итераций бутстрапа.
        alpha (float): Уровень значимости.
        batch_size (int): Размер батча для параллельного выполнения.
        n_jobs (int): Количество параллельных процессов.
        bootstrap_function (function): Функция для вычисления метрики.
        helping_function (function): Вспомогательная функция для обработки выборок.
        arrays_dict (dict): Данные тестовой и контрольной группы.
        additional_information_dict (dict): Дополнительная информация.

    Возвращает:
        PostNormedComparisonResults: Результаты бутстрапа.
    """
    
    array = Parallel(n_jobs=n_jobs)(
        delayed(helping_function)(
            batch_size=batch_size,
            bootstrap_function=bootstrap_function,
            arrays_dict=arrays_dict.copy(),
            additional_information_dict=additional_information_dict.copy(),
            need_estimation=False
        )
        for _ in range(0, num, batch_size)
    )
    res = np.concatenate(array)
    estimation = helping_function(1, bootstrap_function, arrays_dict.copy(), additional_information_dict.copy(), need_estimation=True)
    quantiles = np.quantile(res, [alpha / 2, 1 - alpha / 2])
    delta = quantiles[-1] - quantiles[0]
    pval = 2 * min(np.mean(res < 0), np.mean(res > 0))
    
    return PostNormedComparisonResults(pval, estimation, delta, quantiles[0], quantiles[-1])


def boot_parallel_helper(batch_size, bootstrap_function, arrays_dict, additional_information_dict, need_estimation):
    """
    Вспомогательная функция для выполнения бутстрапа с разбиением выборки.

    Параметры:
        batch_size (int): Размер батча.
        bootstrap_function (function): Функция для расчёта метрики.
        arrays_dict (dict): Данные тестовой и контрольной группы.
        additional_information_dict (dict): Дополнительная информация.
        need_estimation (bool): Флаг для выполнения одной оценки вместо бутстрапа.

    Возвращает:
        float или array: Результаты бутстрапа или единственной оценки.
    """
    
    control_size = len(arrays_dict['after_control'])
    test_size = len(arrays_dict['after_test'])
    if not need_estimation:
        control_inds = sps.poisson(mu=1).rvs((batch_size, control_size))
        test_inds = sps.poisson(mu=1).rvs((batch_size, test_size))
        after_control_sum = np.sum(arrays_dict['after_control'] * control_inds, axis=1)
        before_control_sum = np.sum(arrays_dict['before_control'] * control_inds, axis=1)
        test_after_sum = np.sum(arrays_dict['after_test'] * test_inds, axis=1)
        test_before_sum = np.sum(arrays_dict['before_test'] * test_inds, axis=1)
    else:
        after_control_sum = np.sum(arrays_dict['after_control'])
        before_control_sum = np.sum(arrays_dict['before_control'])
        test_after_sum = np.sum(arrays_dict['after_test'])
        test_before_sum = np.sum(arrays_dict['before_test'])
    
    sum_control_test_dict = {
        'test_after_sum': after_control_sum,
        'test_before_sum': test_before_sum,
        'control_after_sum': after_control_sum,
        'control_before_sum': before_control_sum
    }
    
    return bootstrap_function(sum_control_test_dict, additional_information_dict)


def bootstrap(after_control, before_control, after_test, before_test, alpha=0.05,
              bootstrap_function=bootstrap_main_statistics, num=1000, n_jobs=50, batch_size=100,
              additional_information_dict={}):
    """
    Выполняет бутстрап.

    Параметры:
        after_control, before_control (array): Данные контрольной группы (После / До).
        after_test, before_test (array): Данные тестовой группы (После / До).
        alpha (float): Уровень значимости.
        bootstrap_function (function): Функция для расчёта метрики.
        num (int): Количество итераций бутстрапа.
        n_jobs (int): Количество параллельных процессов.
        batch_size (int): Размер батча.
        additional_information_dict (dict): Дополнительные параметры.

    Возвращает:
        PostNormedComparisonResults: Результаты бутстрапа.
    """
    
    assert len(after_control) == len(before_control)
    arrays_dict = {
        'after_control': after_control,
        'before_control': before_control,
        'after_test': after_test,
        'before_test': before_test
    }
    
    return _bootstrap_body(num, alpha, batch_size, n_jobs, bootstrap_function, boot_parallel_helper, arrays_dict, additional_information_dict)


def bucketisation(sample, buckets_num=200):
    """
    Делит данные на бакеты.

    Параметры:
        sample (array): Входной массив данных.
        buckets_num (int): Количество бакетов.

    Возвращает:
        array: Средние значения по бакетам.
    """
    
    bucket_sum_sample = np.zeros(buckets_num)
    bucket_sample_size_sample = np.zeros(buckets_num)
    for ind, element in enumerate(sample):
        bucket_ind = ind % buckets_num
        bucket_sum_sample[bucket_ind] += element
        bucket_sample_size_sample[bucket_ind] += 1
        
    return bucket_sum_sample / bucket_sample_size_sample


def linearize_metric(num, den):
    """
    Линеаризует метрику отношения двух массивов.

    Параметры:
        num, den (array): Числитель и знаменатель.

    Возвращает:
        array: Линеаризованный массив.
    """
    
    E_num = np.mean(num)
    E_den = np.mean(den)
    lin_metric = E_num / E_den + 1 / E_den * (num - E_num / E_den * den)
    
    return lin_metric


def simple_relative_ttest_CI(test, control, len_T, len_C, alpha, show_tv_group_results):
    """
    Строит доверительный интервал в случае относительного T-test.

    Параметры:
        test, control (array): Данные тестовой и контрольной групп.
        len_T, len_C (int): Размеры выборок теста и контроля.
        alpha (float): Уровень значимости.
        show_tv_group_results (bool): Флаг для вывода результатов.

    Возвращает:
        PostNormedComparisonResults: Результаты теста.
    """
    
    lin_metric = linearize_metric(test, control)
    difference_distribution = sps.norm(loc=np.mean(lin_metric) - 1, scale=sps.sem(lin_metric))

    if show_tv_group_results:
        left_bound, right_bound = difference_distribution.ppf([alpha / 2, 1 - alpha / 2])
        pvalue = 2 * min(difference_distribution.cdf(0), difference_distribution.sf(0))
        
        return PostNormedComparisonResults(pvalue, difference_distribution.mean(), right_bound - left_bound, left_bound, right_bound)

    test_size_sample = sps.poisson(mu=len_T).rvs(10000)
    size_sample = test_size_sample / (test_size_sample + sps.poisson(mu=len_C).rvs(10000))
    effect_sample = difference_distribution.rvs(10000)
    final_sample = size_sample * effect_sample
    left_bound, right_bound = np.quantile(final_sample, [alpha / 2, 1 - alpha / 2])
    pvalue = 2 * min(np.mean(final_sample <= 0), np.mean(final_sample >= 0))
    
    return PostNormedComparisonResults(pvalue, np.mean(final_sample), right_bound - left_bound, left_bound, right_bound)


def post_normed_ttest_for_full_effect(after_control, before_control, after_test, before_test, show_tv_group_results, alpha=0.05):
    """
    Строит доверительный интервал для эффекта теста с пост-нормировкой после бакетизации.

    Параметры:
        after_control, before_control (array): Контрольные данные до и после.
        after_test, before_test (array): Тестовые данные до и после.
        show_tv_group_results (bool): Флаг для вывода результатов.
        alpha (float): Уровень значимости.

    Возвращает:
        PostNormedComparisonResults: Результаты теста.
    """
    control = after_control
    test = after_test
    control_before = before_control
    test_before = before_test
    
    left_bound = 0
    right_bound = 0

    # Выполняем бакетирование.
    bucket_test = bucketisation(test, buckets_num=200)
    bucket_test_before = bucketisation(test_before, buckets_num=200)

    bucket_control = bucketisation(control, buckets_num=200)
    bucket_control_before = bucketisation(control_before, buckets_num=200)

    T_lin = linearize_metric(bucket_test, bucket_test_before)
    C_lin = linearize_metric(bucket_control, bucket_control_before)
    
    return simple_relative_ttest_CI(T_lin, C_lin, len(test), len(control), 
                                    alpha=alpha, show_tv_group_results=show_tv_group_results)
