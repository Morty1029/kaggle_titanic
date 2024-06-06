from Training.ModelRun import ModelRun
from Decorators.patterns import singleton


@singleton
class ModelPathConstructor:
    def __init__(self):
        self._run = None

    def set_model_run(self, model_run):
        self._run = model_run

    def get_model_path(self):
        return f'results\\{self._run.model_type.name}\\models\\model_{self._run.__str__()}'

    def get_params_path(self):
        return f'results\\{self._run.model_type.name}\\params\\params_{self._run.__str__()}'

    def get_metrics_path(self):
        return f'results\\{self._run.model_type.name}\\metrics\\metrics_{self._run.__str__()}'
