import pandas as pd
from pandas.api.types import CategoricalDtype


class ReplaceRare(object):
    """
    This works on pd.Series
    """
    def __init__(self, min_val=50, replace_val='<RARE>'):
        self.min_val = min_val
        self.replace_val = replace_val
        self.is_fit = False
        
    def fit(self, series_train, *args):
        self.val_counts = series_train.value_counts()
        self.is_fit = True
        
    def transform(self, series):
        if not self.is_fit:
            raise Exception('You must fit before you transform')
        rares = self.val_counts[self.val_counts<self.min_val].index.values
        series.loc[series_train.isin(rares)] = self.replace_val
        return series
    
    def fit_transform(self, series_train, series_test=None):
        self.fit(series_train)
        series_train = self.transfrom(series_train)
        if series_test is None:
            return series_train
        else:
            series_test = self.transform(series_test)
            return series_train, series_test
        
        
class EncodeCategories(object):
    """
    This works on ???
    """
    def __init__(self, oov_adjustment=False):
        self.oov_adjustment = oov_adjustment
        self.is_fit = False
    
    def fit(self, series_train, *args):
        self.catDtype = CategoricalDtype(categories=series.value_counts().index.values)
        self.is_fit = True
        
    def transform(self, series):
        if not self.is_fit:
            raise Exception('You must fit before you transform')
        if self.oov_adjustment:
            delta = 1
        else:
            delta = 0
        return series.astype(catDtype).cat.codes.values + delta
    
    def fit_transform(self, series_train, series_test=None):
        self.fit(series_train)
        series_train = self.transfrom(series_train)
        if series_test is None:
            return series_train
        else:
            series_test = self.transform(series_test)
            return series_train, series_test

