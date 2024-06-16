from training.model_utils.ModelTypes import ModelTypes
from training.model_utils.ModelObject import ModelObject
from config_classes.Settings import Settings
from catboost import CatBoostClassifier, CatBoostRegressor
from enums.TaskTypes import TaskTypes
from typing import Union
from dataset_classes.SplitDataset import SplitDataset
from dataset_classes.Dataset import Dataset


class ModelConstructor:

    @staticmethod
    def get_model_of_type(model_type: ModelTypes) -> ModelObject:
        if model_type is not None:
            return ModelObject(model_type.value, model_type)
        else:
            settings = Settings()
            model_type = ModelTypes[settings.model_type]
            print('No ModelType. Model constructor returned model from settings')
            return ModelObject(model_type.value, model_type)

    @staticmethod
    def get_additional_params(model_type: ModelTypes,
                              split_data: Union[SplitDataset, None] = None,
                              dataset: Union[Dataset, None] = None) -> dict:
        if model_type == ModelTypes.CATBOOST:
            return {
                'eval_set': (split_data.x_test, split_data.y_test),
                'early_stopping_rounds': 10,
                'cat_features': dataset.cat_cols
            }
        else:
            # TODO
            return {}

    @staticmethod
    def get_model_from_file(path, model_type: ModelTypes):
        settings = Settings()
        if model_type == ModelTypes.CATBOOST:
            model = CatBoostRegressor() if settings.task_type == TaskTypes.REGRESSION else CatBoostClassifier()
            return model.load_model(fname=path, format='cbm')
        else:
            # TODO
            pass
