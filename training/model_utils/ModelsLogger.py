from training.model_utils.ModelTypes import ModelTypes
from training.model_utils.ModelObject import ModelObject
from training.model_utils.ModelRun import ModelRun
import mlflow


class ModelsLogger:

    @staticmethod
    def model_to_file(model_object: ModelObject, path: str):
        if model_object.model_type == ModelTypes.CATBOOST:
            model_object.model.save_model(path, format='cbm')
        else:
            # TODO
            pass

    @staticmethod
    def model_to_mlflow(model_run: ModelRun):
        if model_run.model_object.model_type == ModelTypes.CATBOOST:
            mlflow.catboost.log_model(model_run.model_object.model, model_run.__str__())
