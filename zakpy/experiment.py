import hashlib
import pandas as pd
import json
from uuid import uuid4


def get_df_hash(df):
    hash_val = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()
    return hash_val


class Experiment(object):
    def __init__(self):
        self.datasets = dict()
        self.models = dict()
        self.description = ""
        self.id = str(uuid4())[:8]
        self.score_file = None
        self.metrics = dict()
    
    def add_dataset(self, name, df):
        self.datasets.update({name: get_df_hash(df)})
        
    def add_model(self, name, model, params):
        self.models.update({name: {'params': params, 'model_type': model.__class__.__name__}})
        
    def add_description(self, description):
        self.description = description
        
    def add_score_file(self, score_file):
        self.score_file = score_file
        
    def add_metrics(self, model_name, metric_name, metrics):
        new_metric = {metric_name: metrics}
        if self.metrics.get(model_name) is not None:
            self.metrics[model_name].update(new_metric)
        else:
            self.metrics[model_name] = new_metric
        
    def set_feature_list(self, feature_list):
        self.feature_list = feature_list
        
    def to_json(self, fp):
        summary = {
            "datasets": self.datasets,
            "models": self.models,
            "description": self.description,
            "all_features": self.feature_list.all_features,
            "target": self.feature_list.target,
            "numeric_features": self.feature_list.all_numeric_cols,
            "categorical_features": self.feature_list.all_categorical_cols,
        }
        if self.score_file:
            summary.update({"score_file": self.score_file})
            
        if self.metrics:
            summary.update({"metrics": self.metrics})
        with open(fp, 'w') as f:
            json.dump(summary, f, indent=4)
    
    def compare_dataset(self, name, df):
        """
        Compares the hash value on record for 'name' to 
        the hash value of the dataset passed in: df
        """
        existing_hash = self.datasets[name]
        new_hash = get_df_hash(df)
        if existing_hash == new_hash:
            return True
        else:
            return False
    