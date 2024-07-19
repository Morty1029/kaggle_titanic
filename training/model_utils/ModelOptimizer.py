from training.model_utils.ModelConstructor import ModelConstructor
from training.model_utils.ModelObject import ModelObject
from dataset_classes.Dataset import Dataset
from dataset_classes.DatasetSplitter import DatasetSplitter
from training.model_utils.ModelTypes import ModelTypes
from optuna.study import create_study
from typing import Callable
from utils.DictLogger import DictLogger


class ModelOptimizer:
    def __init__(self,
                 model_object: ModelObject,
                 dataset: Dataset,
                 metric: Callable,
                 path_to_best_params=None):
        self.model_object = model_object
        self.dataset = dataset
        self.metric = metric
        self.path = path_to_best_params
        self.best_params = {}

    def optimize_model(self):
        model_type = self.model_object.model_type
        study = create_study(direction='maximize')
        optimize_func: Callable
        if model_type == ModelTypes.CATBOOST:
            optimize_func = self.__optimize_catboost
        else:
            # TODO
            optimize_func = self.__optimize_catboost
            pass
        study.optimize(optimize_func, n_trials=2)
        self.best_params = study.best_params
        if self.path is not None:
            DictLogger().dict_to_json(self.best_params, self.path)

    def __optimize_catboost(self, trial):
        params = {
            'iterations': trial.suggest_int('iterations', 10, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
            'depth': trial.suggest_int('depth', 3, 16),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0, 5),

        }
        return self.__trial(params)

    def __trial(self, params):
        split_data = DatasetSplitter(self.dataset).split_dataset(True)
        model = self.model_object.model.copy()
        model.set_params(**params)
        additional_params = ModelConstructor.get_additional_params(self.model_object.model_type,
                                                                   self.dataset)
        model.fit(split_data.x_train,
                  split_data.y_train,
                  **additional_params)
        predictions = model.predict(split_data.x_val)

        result = self.metric(split_data.y_val, predictions)

        return result
