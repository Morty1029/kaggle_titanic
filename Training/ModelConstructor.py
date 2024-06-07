from Training.ModelTypes import ModelTypes
from Training.ModelObject import ModelObject
from ConfigsClasses.Settings import Settings


class ModelConstructor:

    @staticmethod
    def get_model(model_type: ModelTypes) -> ModelObject:
        if model_type is not None:
            if isinstance(model_type, ModelTypes):
                return ModelObject(model_type.value, model_type)
            else:
                print('ModelType is not from ModelTypes. Model constructor returned model from settings')
        else:
            print('No ModelType. Model constructor returned model from settings')
        settings = Settings()
        model_type = ModelTypes[settings.model_type]
        return ModelObject(model_type.value, model_type)

    @staticmethod
    def get_additional_params(model_type: ModelTypes, split_data=None, dataset=None) -> dict:
        if model_type == ModelTypes.CATBOOST:
            return {
                'eval_set': (split_data.x_test, split_data.y_test),
                'early_stopping_rounds': 10,
                'cat_features': dataset.cat_cols
            }
        else:
            # TODO
            return {}
