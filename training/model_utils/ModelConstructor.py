from training.model_utils.ModelTypes import ModelTypes
from training.model_utils.ModelObject import ModelObject
from config_classes.Settings import Settings
from catboost import CatBoostClassifier, CatBoostRegressor
from enums.TaskTypes import TaskTypes
from utils.PathConstructor import PathConstructor
from dataset_classes.SplitDataset import SplitDataset
from dataset_classes.Dataset import Dataset
from typing import Union
from utils.FileReader import FileReader


class ModelConstructor:

    @staticmethod
    def get_model_of_type(model_type: ModelTypes = None) -> ModelObject:
        if model_type is not None:
            return ModelObject(model_type.value, model_type)
        else:
            settings = Settings()
            model_type = settings.model_type
            print('No ModelType. Model constructor returned model from settings')
            return ModelObject(model_type.value, model_type)

    @staticmethod
    def get_additional_params(model_type: ModelTypes,
                              dataset: Union[Dataset, None] = None) -> dict:
        if model_type == ModelTypes.CATBOOST:
            main_params = {}
            if dataset is not None:
                main_params = main_params | {'cat_features': dataset.cat_cols}
            return main_params
        else:
            # TODO
            return {}

    @staticmethod
    def get_model_from_file(path, model_type: Union[ModelTypes, None] = None) -> ModelObject:
        settings = Settings()
        if model_type is None:
            model_type = settings.model_type
        if model_type == ModelTypes.CATBOOST:
            model = CatBoostRegressor() if settings.task_type == TaskTypes.REGRESSION else CatBoostClassifier()
            model.load_model(fname=path, format='cbm')
            return ModelObject(model=model,
                               model_type=model_type)
        else:
            # TODO
            pass

    @staticmethod
    def get_model_from_settings() -> ModelObject:
        model_path = PathConstructor.get_model_path_from_settings()
        model = ModelConstructor.get_model_from_file(model_path)
        return model

    @staticmethod
    def get_params_from_settings() -> dict:
        params_path = PathConstructor.get_metrics_path_from_settings()
        params = FileReader.read_json(params_path)
        return params
