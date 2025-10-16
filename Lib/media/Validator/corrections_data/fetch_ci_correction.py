import sys
lib_dir = '/srv/data/my_shared_data/Lib'
if lib_dir not in sys.path:
    sys.path.append(lib_dir)
from media.Validator.corrections_data.ci_corrections import *

def fetch_ci_correction(model, **kwargs):
    models = {'YoY', 'RegAB'}
    if model not in models:
        raise ValueError('Invalid model name')
    if model == 'YoY':
        assert 'metric' in kwargs 
        assert 'alpha' in kwargs
        assert 'learning_period' in kwargs
        assert 'interval_length' in kwargs
        assert 'by_week' in kwargs
        for ci_corr_data in YOY_ci_corrections:
            if (kwargs['metric'] == ci_corr_data['metric']) and \
               (kwargs['alpha'] == ci_corr_data['alpha']) and \
               (kwargs['learning_period'] == ci_corr_data['learning_period']) and \
               (kwargs['interval_length'] == ci_corr_data['interval_length']) and \
               (kwargs['by_week'] == ci_corr_data['by_week']):
                return ci_corr_data['correction']
    return 1
    