import numpy as np
import pandas as pd
from scipy.optimize import brentq
from statsmodels.stats.proportion import proportion_confint

class ConfIntervalCorrection():
    
    
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed=seed)
        
    
    def _split_data(self, val_results_input, split_by_recency, frac):
        val_results = val_results_input.copy()
        slice_labels = val_results['label'].unique()
        val_results['is_test'] = False
        for slice_label in slice_labels:
            val_results_in_slice = val_results.query('label == @slice_label')
            n = len(val_results_in_slice)
            num_tests = max(1, round((1 - frac) * n))
            split = np.array([False] * (n - num_tests) + [True] * num_tests)
            if split_by_recency:
                sorted_index = pd.to_datetime(val_results_in_slice['start_date'], format='%Y-%m-%d').sort_values().index
                val_results.loc[sorted_index, 'is_test'] = split
            else:
                self.rng.shuffle(split)
                val_results.loc[val_results_in_slice.index, 'is_test'] = split
        self.val_results = val_results
        return val_results
        
           
    def calculate_fpr(self, corr_coef, val_results, slice_label, is_test):
        if slice_label is not None:
            df = val_results.query('is_test == @is_test and label == @slice_label')
        else:
            df = val_results.query('is_test == @is_test')
        ci_width = df['effect right'] - df['effect left']
        right_corrected = df['effect'] + (df['effect right'] - df['effect']) * corr_coef
        left_corrected = df['effect'] - (df['effect'] - df['effect left']) * corr_coef
        fpr = ((left_corrected > 0) | (right_corrected < 0)).sum() / len(df)
        return fpr
    
    
    def calculate_total_fpr(self, corr_coef, val_results):
        df = val_results
        ci_width = df['effect right'] - df['effect left']
        right_corrected = df['effect'] + (df['effect right'] - df['effect']) * corr_coef
        left_corrected = df['effect'] - (df['effect'] - df['effect left']) * corr_coef
        fpr = ((left_corrected > 0) | (right_corrected < 0)).sum() / len(df)
        return fpr
    
    
    def calculate_fpr_synt(self, corr_coef, val_results_synt_effects, slice_label, synt_effect):
        if slice_label:
            df = val_results_synt_effects.query('label == @slice_label')
        else:
            df = val_results_synt_effects
        ci_width = df['effect right'] - df['effect left']
        right_corrected = df['effect'] + (df['effect right'] - df['effect']) * corr_coef
        left_corrected = df['effect'] - (df['effect'] - df['effect left']) * corr_coef
        fpr = ((synt_effect - 1 < left_corrected) | (right_corrected < synt_effect - 1)).sum() / len(df)
        return fpr
    
    
    def calculate_mean_ci_width(self, val_results, slice_label, is_test):
        if slice_label is not None:
            df = val_results.query('is_test == @is_test and label == @slice_label')
        else:
            df = val_results.query('is_test == @is_test')
        return (df['effect right'] - df['effect left']).mean()
    
    
    def calculate_total_mean_ci_width(self, val_results, slice_label):
        if slice_label is not None:
            df = val_results.query('label == @slice_label')
        else:
            df = val_results
        return (df['effect right'] - df['effect left']).mean()
        
        
    def _calculate_correction(self, val_results, alpha,
                             min_correction = 0.33, max_correction = 3):
        func_to_opt = lambda x, *args: self.calculate_fpr(x, *args) - alpha
        print(func_to_opt(min_correction, val_results, None, False), func_to_opt(max_correction, val_results, None, False))
        corr_coef = brentq(func_to_opt, min_correction, max_correction, 
                           args = (val_results, None, False))
        #if self.calculate_fpr(corr_coef, val_results, slice_label=None, is_test=False) - alpha > 0.005:
        #    raise ValueError('Failed to find correction coefficient')
        return corr_coef 
        
        
    def calculate_fpr_ci(self, p_hat, n):
        n_s = round(p_hat * n)
        n_f = n - n_s
        l, r = proportion_confint(n_s, n, alpha=0.05, method='wilson')    
        return l, r
        
        
    def create_correction_report(self, val_results_input, 
                             alpha, split_by_recency=False, frac=0.8, min_correction = 0.33, max_correction = 3):
        val_results = self._split_data(val_results_input, split_by_recency, frac)
        corr_coef = self._calculate_correction(val_results, alpha, min_correction, max_correction)
        slice_labels = list(val_results['label'].unique()) + ['Total']
        test_fprs = dict()
        for slice_label in slice_labels:
            if slice_label == 'Total':
                n_test = len(val_results.query('is_test == True'))
                n_train = len(val_results.query('is_test == False'))
                test_fpr_init = self.calculate_fpr(1, val_results, None, is_test=True)
                test_fpr_corr = self.calculate_fpr(corr_coef, val_results, None, is_test=True)
                train_fpr_init = self.calculate_fpr(1, val_results, None, is_test=False)
                train_fpr_corr = self.calculate_fpr(corr_coef, val_results, None, is_test=False)
                ci_width_test = self.calculate_mean_ci_width(val_results, None, is_test=True)
                ci_width_train = self.calculate_mean_ci_width(val_results, None, is_test=False)
                
                test_fprs[slice_label] = [test_fpr_init, *self.calculate_fpr_ci(test_fpr_init, n_test),
                                          test_fpr_corr, *self.calculate_fpr_ci(test_fpr_corr, n_test),
                                          ci_width_test,
                                          n_test, 
                                          train_fpr_init, *self.calculate_fpr_ci(train_fpr_init, n_train),
                                          train_fpr_corr, *self.calculate_fpr_ci(train_fpr_corr, n_train),
                                          ci_width_train,
                                          n_train]
            else:
                n_test = len(val_results.query('label == @slice_label and is_test == True'))
                n_train = len(val_results.query('label == @slice_label and is_test == False'))
                test_fpr_init = self.calculate_fpr(1, val_results, slice_label, is_test=True)
                test_fpr_corr = self.calculate_fpr(corr_coef, val_results, slice_label, is_test=True)
                train_fpr_init = self.calculate_fpr(1, val_results, slice_label, is_test=False)
                train_fpr_corr = self.calculate_fpr(corr_coef, val_results, slice_label, is_test=False)
                ci_width_test = self.calculate_mean_ci_width(val_results, slice_label, is_test=True)
                ci_width_train = self.calculate_mean_ci_width(val_results, slice_label, is_test=False)
                
                test_fprs[slice_label] = [test_fpr_init, *self.calculate_fpr_ci(test_fpr_init, n_test),
                                          test_fpr_corr, *self.calculate_fpr_ci(test_fpr_corr, n_test),
                                          ci_width_test,
                                          n_test, 
                                          train_fpr_init, *self.calculate_fpr_ci(train_fpr_init, n_train),
                                          train_fpr_corr, *self.calculate_fpr_ci(train_fpr_corr, n_train),
                                          ci_width_train,
                                          n_train]
        
        report_df = pd.DataFrame.from_dict(test_fprs, orient='index', columns=[
            'test fpr initial', 'test fpr init ci low', 'test fpr init ci high', 
            'test fpr with correction', 'test fpr with corr ci low', 'test fpr with corr ci high',
            'ci width test',
            'test size',
            'train fpr initial', 'train fpr init ci low', 'train fpr init ci high',
            'train fpr with correction', 'train fpr with corr ci low', 'train fpr with corr ci high', 
            'ci width train',
            'train size'])
        
        
        return self.prettify_report(corr_coef, report_df)
    
    
    def create_fpr_synt_report(self, val_results_synt, alpha, synt_effect, corr_coef=1):
        slice_labels = list(val_results_synt['label'].unique()) + ['Total']
        test_fprs = dict()
        for slice_label in slice_labels:
            if slice_label == 'Total':
                n_test = len(val_results_synt)
                fpr_synt_init = self.calculate_fpr_synt(1, val_results_synt, None,  synt_effect)
                fpr_synt_corr = self.calculate_fpr_synt(corr_coef, val_results_synt, None, synt_effect)
                ci_width = self.calculate_total_mean_ci_width(val_results_synt, None)
                test_fprs[slice_label] = [fpr_synt_init, *self.calculate_fpr_ci(fpr_synt_init, n_test),
                                          fpr_synt_corr, *self.calculate_fpr_ci(fpr_synt_corr, n_test),
                                          ci_width, n_test]
            else:
                n_test = len(val_results_synt.query('label == @slice_label'))
                fpr_synt_init = self.calculate_fpr_synt(1, val_results_synt, slice_label, synt_effect)
                fpr_synt_corr = self.calculate_fpr_synt(corr_coef, val_results_synt, slice_label, synt_effect)
                ci_width = self.calculate_total_mean_ci_width(val_results_synt, slice_label)
                test_fprs[slice_label] = [fpr_synt_init, *self.calculate_fpr_ci(fpr_synt_init, n_test),
                                          fpr_synt_corr, *self.calculate_fpr_ci(fpr_synt_corr, n_test),
                                          ci_width, n_test]
        report_synt = pd.DataFrame.from_dict(test_fprs, orient='index', 
                                      columns=['test fpr initial', 'test fpr init ci low', 'test fpr init ci high', 
                                               'test fpr with correction', 'test fpr corr ci low', 'test fpr corr ci high',
                                               'ci_width', 'test size'])
       
        return self.prettify_synt_report(corr_coef, report_synt)
    
    
    def prettify_report(self, corr_coef, report):
        test_fpr_init_col = report.apply(lambda x: f"{x['test fpr initial']:.1%} <br> \
        [{x['test fpr init ci low']:.1%}; {x['test fpr init ci high']:.1%}]", axis=1)

        test_fpr_corr_col = report.apply(lambda x: f"{x['test fpr with correction']:.1%} <br> \
        [{x['test fpr with corr ci low']:.1%}; {x['test fpr with corr ci high']:.1%}]", axis=1)

        train_fpr_init_col = report.apply(lambda x: f"{x['train fpr initial']:.1%} <br> \
        [{x['train fpr init ci low']:.1%}; {x['train fpr init ci high']:.1%}]", axis=1)

        train_fpr_corr_col = report.apply(lambda x: f"{x['train fpr with correction']:.1%} <br> \
        [{x['train fpr with corr ci low']:.1%}; {x['train fpr with corr ci high']:.1%}]", axis=1)

        ci_width_test = report['ci width test'].apply(lambda x: f"{x:.1%}") 
        ci_width_test_corr = report['ci width test'].apply(lambda x: f"{corr_coef * x:.1%}")

        ci_width_train = report['ci width train'].apply(lambda x: f"{x:.1%}") 
        ci_width_train_corr = report['ci width train'].apply(lambda x: f"{corr_coef * x:.1%}")



        report_pretty = pd.DataFrame([report['test size'], test_fpr_init_col, test_fpr_corr_col, 
                                      ci_width_test, ci_width_test_corr,
                                      report['train size'], train_fpr_init_col, train_fpr_corr_col,
                                      ci_width_train, ci_width_train_corr],
                                      index=['test size', 'test fpr initial (with 95% ci)',
                                            'test fpr corrected (with 95% ci)', 'ci width test',
                                            'ci width test corrected',
                                            'train size', 'train fpr initial (with 95% ci)', 
                                            'train fpr corrected (with 95% ci)', 'ci width train',
                                            'ci width train corrected',]).transpose().style.set_table_styles([
                {
                "selector" :"td",
                "props": "text-align: center; border-style: solid"
                },
                {
                "selector" :"th.col_heading",
                "props": "text-align: center; border-style: solid"
                },
                {
                "selector" :"th.row_heading",
                "props": "text-align: center; border-style: solid"
                }
            ])
        
        corr_coef_dict = {
        'correction coef' : f'{corr_coef:.2f}'
        }
        corr_coef_pretty = pd.DataFrame.from_dict(corr_coef_dict, orient='index', columns=['value'])
        return corr_coef_pretty, report_pretty
    
    
    def prettify_synt_report(self, corr_coef, report_synt):
        test_fpr_init_col = report_synt.apply(lambda x: f"{x['test fpr initial']:.1%} <br> \
        [{x['test fpr init ci low']:.1%}; {x['test fpr init ci high']:.1%}]", axis=1)

        test_fpr_corr_col = report_synt.apply(lambda x: f"{x['test fpr with correction']:.1%} <br> \
        [{x['test fpr corr ci low']:.1%}; {x['test fpr corr ci high']:.1%}]", axis=1)

        ci_width_init = report_synt['ci_width'].apply(lambda x: f"{x:.1%}")

        ci_width_corr = report_synt['ci_width'].apply(lambda x: f"{corr_coef * x:.1%}")

        report_synt_pretty = pd.DataFrame([report_synt['test size'], test_fpr_init_col, #test_fpr_corr_col, 
                                           ci_width_init], #ci_width_corr], 
                                           index=[ 'test size', 'test fpr initial (with 95% ci)', #'test fpr corrected (with 95% ci)', 
                                                  'ci width init']).transpose().\
            style.set_table_styles([ 
                {
                "selector" :"td",
                "props": "text-align: center; border-style: solid"
                },
                {
                "selector" :"th.col_heading",
                "props": "text-align: center; border-style: solid"
                },
                {
                "selector" :"th.row_heading",
                "props": "text-align: center; border-style: solid"
                }
            ])

        return report_synt_pretty


