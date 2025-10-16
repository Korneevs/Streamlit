"""
Set of configuration helper functions.
"""

__version__ = '0.1.3'

__all__ = [
    'configure_logger',
    'log_exception',
    'retry',
    'SHORT_FMT',
    'DEFAULT_DATE_FMT',
    'DEFAULT_FMT',
    'log_to_statsd',
    'log_time',
    'Timer',
    'LogTimer',
]

import datetime
import logging
import logging.handlers
import os.path as path
import itertools
from functools import wraps
import time
from timeit import default_timer


SHORT_FMT = '[%(asctime)s] %(levelname)5s :: %(message)s'
DEFAULT_FMT = '[%(asctime)s] %(name)12s %(process)4d %(levelname)5s :: %(message)s'
DEFAULT_DATE_FMT = '%Y-%m-%d %H:%M:%S'
# TODO: to be done later
COLORED_FMT = '[%(asctime)s] %(name)12s %(process)4d %(levelname)5s :: %(message)s'


def configure_logger(
        logger_name='log',
        log_dir='/tmp/',
        to_stdout=True,
        to_file=True,
        level=logging.DEBUG,
        log_format=DEFAULT_FMT,
        date_format=DEFAULT_DATE_FMT,
        filename='',
        verbose=True,
) -> logging.Logger:
    """
    :return: Fully configured logger with given name and stdout/file output with pretty standard handler.

    Filename to log automatically derived from logger_name and directory is taken from parameters.
    This function clears all previous handlers for this logger name.

    Usage
    ------
    ```python
    from log_helper import configure_logger
    logger = configure_logger('myprog')
    logger.info('hello')

    # [2017-07-22 14:36:53]       myprog 3508  INFO :: hello
    # [2017-07-22 14:36:53]  myprog.log2 3508 DEBUG :: hello there too
    ```

    Note that default formatter allows to easily distinguish between
    - different loggers writing to same file
    - different runs (or workers from different processes) writing to same file

    and
    - visually separate message from context

    while maintaining clear and compact console view.
    """

    if not filename:
        filename = logger_name + '.log'
    log_location = path.join(log_dir, filename)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.handlers = []

    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Stdout
    if to_stdout:
        stdout = logging.StreamHandler()
        stdout.setLevel(level)
        stdout.setFormatter(formatter)
        logger.addHandler(stdout)

    # Filehandler
    if to_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_location,
            mode='a',
            maxBytes=10**8,
            backupCount=1,
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        if verbose:
            print('Log to file {}'.format(log_location))

    logger.propagate = False
    return logger


def log_exception(logger_name='log',
                  on_exception=lambda *args: None,
                  exception=Exception,
                  consume_exception=False):
    """
    Decorator, logs exception to logger_name.
    """
    logger = logging.getLogger(logger_name)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception as problem:
                logger.exception('Something went wrong in function {}'.format(func.__name__))
                on_exception(problem)
                if not consume_exception:
                    raise
        return wrapper

    return decorator


def retry(exception=Exception,
          report=lambda *args: None,
          delays_seconds=(60, 300, 600, 1200, 2400, 3600)):
    """
    Decorator for retrying decorated function if specified exception occurs with specified delays.

    Usage:
    ```python
    @retry(exception=pyodbc.Error, report=print, delays_seconds=(1, 2)))
    def do_some_sql():
        raise pyodbc.Error


    res = do_some_sql()
    # retryable failed: pyodbc.Error ... -- delaying for 1 sec"
    # retryable failed: pyodbc.Error ... -- delaying for 1 sec"
    # retryable failed definitely pyodbc.Error
    # exception raised!

    # And yes, we can log error to logger.
    @retry(report=logger.error)
    def do_some_sql():
        raise pyodbc.Error
    ```
    """

    def wrapper(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            problems = []
            for delay in itertools.chain(delays_seconds, [None]):
                try:
                    return func(*args, **kwargs)
                except exception as problem:
                    problems.append(problem)
                    if delay is None:
                        report("retryable failed definitely: {}".format(problems))
                        raise
                    else:
                        report("retryable failed: {} -- delaying for {} sec".format(problem, delay))
                        time.sleep(delay)

        return wrapped

    return wrapper


def log_to_statsd(metric: str,
                  value: float,
                  prefix: str = 'apps.services',
                  host: str = 'aggregator01',
                  port: int = 8126):

    now = datetime.datetime.now()

    try:
        import statsd
        statsd_client = statsd.StatsClient(host, port, prefix)
        statsd_client.gauge(metric, value)
    except OSError:
        print('statsd log: {} Cant log to statsd'.format(now))
    except ImportError:
        print('cant import statsd')


class Timer:

    def __init__(self, name='', verbose=True):
        self.verbose = verbose
        self.timer = default_timer
        self.seconds = .0
        self.name = name

    def __enter__(self):
        self.start = self.timer()
        return self

    def __exit__(self, *args):
        end = self.timer()
        self.seconds = end - self.start
        if self.verbose:
            self.report()

    def report(self):
        print(self.get_message())

    def get_message(self) -> str:
        if self.name:
            return '{} done in {:.1f} s'.format(self.name, self.seconds)
        else:
            return 'done in {:.1f} s'.format(self.seconds)


class LogTimer(Timer):

    def __init__(self, logger=None, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger

    def report(self):
        if self.logger is None:
            super().report()
        else:
            self.logger.info(self.get_message())


log_time = LogTimer