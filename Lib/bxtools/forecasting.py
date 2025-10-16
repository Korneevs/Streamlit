import numpy as np
np.seterr(all='raise')

import pandas as pd
import datetime
import itertools

from bxtools.vertica import VerticaEngine
from bxtools.utils import filter_df, define_logger
import logging

import matplotlib.pyplot as plt

import fbprophet
from scipy import stats


class Prophet(fbprophet.Prophet):
    def __init__(
        self,
        df,
        dim_iter,
        cap_coef,
        forecast_start_ds,
        forecast_end_ds,
        boxcox,
        **kwargs
    ):
        super().__init__(
            **kwargs
        )
        
        self.growth = kwargs.get('growth')
        if forecast_start_ds is not None:
            self.df = df[df.ds < forecast_start_ds].copy()
        else:
            self.df = df.copy()
        
        self.forecast_start_ds = (self.df.ds.max() + datetime.timedelta(days=1))
        
        if self.growth == 'logistic':
            self.cap = self.df.y.iloc[-28:].median() * cap_coef
            self.df['cap'] = self.cap
            self.cap_coef = cap_coef
        
        
        self.forecast_params = dim_iter.copy()
        self.forecast_params.update({
            'cap_coef': cap_coef,
            'forecast_start_ds': self.forecast_start_ds,
            'forecast_end_ds': forecast_end_ds
        })
        
        if boxcox:
            self.df['y_orig'] = self.df.y
            while True:
                try:
                    self.df.y, self.boxcox_lmbda = stats.boxcox(self.df.y_orig)
                except:
                    self.df.drop(self.df.index[0], inplace=True)
                    #print(self.df.shape)
                    continue
                break
            self.forecast_params.update({'boxcox_lmbda': self.boxcox_lmbda})
            
            if self.growth == 'logistic':
                self.df.cap = stats.boxcox(self.df.cap, self.boxcox_lmbda)
                self.cap = stats.boxcox(self.cap, self.boxcox_lmbda)
                print(self.cap)
        
        if forecast_end_ds is not None:
            self.periods = (forecast_end_ds - self.df.ds.max()).days
        else:
            self.periods = 0
        
        self.forecast_params.update(kwargs)
        self.forecast_params.pop('holidays', None)
        self.forecast_params.pop('changepoints', None)
        
        self.dim_iter = dim_iter
        
        
    def invboxcox(self, y):
        if not hasattr(self, 'boxcox_lmbda'):
            raise Exception('boxcox_lmbda is undefined')
        elif self.boxcox_lmbda == 0:
            return(np.exp(y))
        else:
            return np.exp(np.log(self.boxcox_lmbda * y + 1) / self.boxcox_lmbda)
    
    def fit(self, **kwargs):
        
        df_to_fit = self.df.copy()
       
        super().fit(df_to_fit, **kwargs)
        
        #k = 0
        #while self.params.get('k_s').std() < 1e-2:
        #    last_y_index = df_to_fit.ds == df_to_fit[~df_to_fit.y.isnull()].ds.max()
        #    df_to_fit.loc[last_y_index, 'y'] = None
        #    
        #    self.history = None
        #    self.params = {}
        #    k += 1
        #    print(k)
        #    super().fit(df_to_fit, **kwargs)
            
        return self
    
    def predict(self):
        
        fut = super().make_future_dataframe(periods=self.periods)
        
        if self.growth == 'logistic':
            fut['cap'] = self.cap
                
        self.df_forecast = super().predict(fut)

        
        return self
    
    def set_changepoints(self):
        """Set changepoints

        Sets m$changepoints to the dates of changepoints. Either:
        1) The changepoints were passed in explicitly.
            A) They are empty.
            B) They are not empty, and need validation.
        2) We are generating a grid of them.
        3) The user prefers no changepoints be used.
        """
        if self.changepoints is not None:
            if len(self.changepoints) == 0:
                pass
            else:
                too_low = min(self.changepoints) < self.history['ds'].min()
                too_high = max(self.changepoints) > self.history['ds'].max()
                if too_low or too_high:
                    raise ValueError(
                        'Changepoints must fall within training data.')
        else:
            # Place potential changepoints evenly through first 80% of history
            hist_size = np.floor(self.history.shape[0] * 0.99)
            if self.n_changepoints + 1 > hist_size:
                self.n_changepoints = hist_size - 1
                logger.info(
                    'n_changepoints greater than number of observations.'
                    'Using {}.'.format(self.n_changepoints)
                )
            if self.n_changepoints > 0:
                cp_indexes = (
                    np.linspace(0, hist_size, self.n_changepoints + 1)
                    .round()
                    .astype(np.int)
                )
                self.changepoints = (
                    self.history.iloc[cp_indexes]['ds'].tail(-1)
                )
            else:
                # set empty changepoints
                self.changepoints = []
        if len(self.changepoints) > 0:
            self.changepoints_t = np.sort(np.array(
                (self.changepoints - self.start) / self.t_scale))
        else:
            self.changepoints_t = np.array([0])  # dummy changepoint

    def plot_fact(self):
        plt.plot(self.df.ds, self.df.y)
        title='_'.join(list(str(e) for e in self.dim_iter.values()))
        plt.title(title)
        #plt.savefig(filename='figs/{}.png'.format(title), dpi=150)
        plt.show()
        return self
        
    def plot(self, param_list=None):
        if param_list is None:
            param_list = self.forecast_params.keys()
        print(pd.Series(self.forecast_params)[param_list])

        super().plot(self.df_forecast)
        plt.show()
        return self
    
    def plot_components(self, param_list=None):
        if param_list is None:
            param_list = self.forecast_params.keys()
        print(pd.Series(self.forecast_params)[param_list])

        super().plot_components(self.df_forecast)
        plt.show()
        return self
        

class ProphetForecasts(object):
    def __init__(
        self,
        data,
        logger='prophet_forecasts',
    ):
        self.logger = define_logger(logger=logger, level=logging.INFO)
        
        self.data = data.copy()
        self.data['y_orig'] = self.data.y
        
        self.dim_cols = list(e for e in self.data.columns if e not in ['ds', 'y', 'y_orig', 'cap'])
        
        self.dim_iters = self.data.groupby(self.dim_cols).size().reset_index()[self.dim_cols]\
            .sort_values(by=(self.dim_cols), ascending=False)
    
    
    def make_forecasts(
        self,
        dim_filter=None,
        **kwargs
    ):
        for k in kwargs:
            if isinstance(kwargs[k], list):
                continue
            else:
                kwargs[k] = [kwargs[k]]
        
        param_cols = list(kwargs.keys())
        
        iters = []
        
        if dim_filter is None:
            diters = self.dim_iters
        else:
            diters = filter_df(self.dim_iters, dim_filter)
        
        
        for params in itertools.product(*list(kwargs.values())):
            for c, itr in diters.iterrows():
                itr_full = itr.values.tolist() + list(params)
                iters.append(itr_full)

        self.forecast_iters = pd.DataFrame(iters, columns=(self.dim_cols + param_cols))
        
        self.logger.info('{} iters to go'.format(self.forecast_iters.shape[0]))
        
        self.forecasts = []
        for i, itr in self.forecast_iters.iterrows():
            
            dim_iter = itr[self.dim_cols]
            
            df = filter_df(self.data, dim_iter)[['ds', 'y']].copy()
            
            params = itr[param_cols].to_dict()
            
            pf = Prophet(df=df, dim_iter=dim_iter.to_dict(), **params).fit().predict()
            
            self.forecasts.append(pf)
            
            self.logger.info('{i} {itr}'.format(i=i, itr=pf.forecast_params))
            
        self.logger.info('all done'.format(i=i, itr=itr.to_dict()))
        return self
    
    @staticmethod
    def __fix_level_change(df, ds, n_examples=2, step=7):
        
        if df.ds.unique().shape[0] > df.shape[0]:
            raise Exception("'ds' is not unuque!")
        
        df.loc[:, 'y'] = np.log(df.loc[:, 'y'])
        
        left_slc = df.ds < ds
        right_slc = df.ds > ds

        x = ds.toordinal()
        xs = df.loc[:, 'ds'].apply(lambda x: x.toordinal())

        left_examples = df[((xs % step) == (x % step)) & left_slc][-n_examples:]
        right_examples = df[((xs % step) == (x % step)) & right_slc][:n_examples]

        shift = right_examples.y.median() - left_examples.y.median()

        df.loc[left_slc, 'y'] += shift
        df.loc[df.ds==ds, 'y'] += shift / 2
        
        df.loc[:, 'y'] = np.exp(df.loc[:, 'y'])

        return df
    
    @staticmethod
    def __fix_outliers(df, ds, n_examples=4, step=7, treshold=np.exp(0.3)):
        #print(n_examples, step, treshold)
        
        if df.ds.unique().shape[0] > df.shape[0]:
            raise Exception("'ds' is not unuque!")
        
        df.loc[:, 'y'] = np.log(df.loc[:, 'y'])
        treshold = np.log(treshold)
        
        left_slc = df.ds < ds
        right_slc = df.ds > ds

        x = ds.toordinal()
        xs = df.loc[:, 'ds'].apply(lambda x: x.toordinal())
        
        left_examples = df[((xs % step) == (x % step)) & left_slc][-n_examples:]
        right_examples = df[((xs % step) == (x % step)) & right_slc][:n_examples]

        b = pd.concat([left_examples, right_examples]).y.median()

        shift = b - df.loc[df.ds==ds, 'y'].mean()

        if -shift >= treshold:
            df.loc[df.ds==ds, 'y'] += shift
        
        df.loc[:, 'y'] = np.exp(df.loc[:, 'y'])
        
        return df

    def fix_data(self, to_fix, fix_type, **kwargs):
        if fix_type == 'outliers':
            def func(df, ds):
                return self.__fix_outliers(df, ds, **kwargs)
        elif fix_type == 'level':
            def func(df, ds):
                return self.__fix_level_change(df, ds, **kwargs)
            
        dcols = list(e for e in to_fix.columns if e != 'ds')
        dftr = to_fix.groupby(dcols).size().reset_index()[dcols]
        
        diters = filter_df(self.dim_iters, dftr)
        
        for i, ditr in diters.iterrows():
            ix = (self.data[self.dim_cols] == ditr[self.dim_cols]).all(axis=1)
            df = filter_df(self.data, ditr[self.dim_cols])[['ds', 'y']]
            if df.shape[0] == 0:
                self.logger.warning('df is empty')
                continue
            dss = to_fix.loc[(to_fix[dcols] == ditr[dcols]).all(axis=1), 'ds']
            for ds in dss:
                self.data.loc[ix, ['y']] = func(df, ds).loc[:, 'y']

        return self
    
    def plot_differences(self, log_transform=True):
        for i, itr in self.dim_iters.iterrows():
            df = self.data[(self.data[self.dim_cols] == itr[self.dim_cols]).all(axis=1)]
            if df.y.sum() == df.y_orig.sum():
                continue
            y_orig = np.log(df.y_orig) if log_transform else df.y_orig
            y = np.log(df.y) if log_transform else df.y
            plt.plot(df.ds, y_orig)
            plt.plot(df.ds, y)
            title='_'.join(list(str(e) for e in itr[self.dim_cols]))
            plt.title(title)
            #plt.savefig(filename='figs/{}.png'.format(title), dpi=150)
            plt.show()
    
    def plot_all_dims(self, log_transform=True):
        for i, itr in self.dim_iters.iterrows():
            df = filter_df(self.data, itr)
            y = np.log(df.y) if log_transform else df.y
            plt.plot(df.ds, y)
            title='_'.join(list(str(e) for e in itr[self.dim_cols]))
            plt.title(title)
            #plt.savefig(filename='figs/{}.png'.format(title), dpi=150)
            plt.show()
            
    def merge_forecasts(self):
        dffs = []
        for f in self.forecasts:
            dff = f.df_forecast.copy()
            for c in ['', '_lower', '_upper']:
                dff['yfct'+c] = f.df_forecast['yhat'+c].apply(f.invboxcox)
            for k, v in f.forecast_params.items():
                dff[k] = v
            dffs.append(dff)
        self.data_forecast = pd.concat(dffs)
        return self
            
    def export_to_vertica(self, con_vertica, tablename='saef.buyerx_prophet_forecasts'):
        
        select_query = """
        select  *
        from    {t}
        where   False
        """.format(t=tablename)
        
        target_columns = con_vertica.select(select_query).columns
        
        data_to_export = self.data_forecast.copy()
        
        data_to_export['insert_datetime'] = datetime.datetime.now()
        
        for c in target_columns:
            if c not in data_to_export:
                data_to_export[c] = None
        
        con_vertica.insert(data_to_export, tablename, target_columns)
        
        return self