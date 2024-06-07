from Training.ModelRun import ModelRun


class ModelPathConstructor:

    @staticmethod
    def get_model_path(run: ModelRun) -> str:
        return f'results\\{run.model_object.model_type.name}\\models\\model_{run.__str__()}'

    @staticmethod
    def get_params_path(run: ModelRun) -> str:
        return f'results\\{run.model_object.model_type.name}\\params\\params_{run.__str__()}'

    @staticmethod
    def get_metrics_path(run: ModelRun) -> str:
        return f'results\\{run.model_object.model_type.name}\\metrics\\metrics_{run.__str__()}'
