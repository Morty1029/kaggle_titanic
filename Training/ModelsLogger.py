from Training.ModelTypes import ModelTypes
from Training.ModelObject import ModelObject


class ModelsLogger:

    @staticmethod
    def model_to_file(model_object: ModelObject, path: str):
        if model_object.model_type == ModelTypes.CATBOOST:
            model_object.model.save_model(path, format='cbm')
        else:
            # TODO
            pass
