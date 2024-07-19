from training.model_utils.ModelRun import ModelRun


class PathConstructor:

    @staticmethod
    def get_model_path(run: ModelRun) -> str:
        return f'results\\{run.model_object.model_type.name}\\models\\model_{run.__str__()}'

    @staticmethod
    def get_params_path(run: ModelRun) -> str:
        return f'results\\{run.model_object.model_type.name}\\params\\params_{run.__str__()}'

    @staticmethod
    def get_metrics_path(run: ModelRun) -> str:
        return f'results\\{run.model_object.model_type.name}\\metrics\\metrics_{run.__str__()}'

    @staticmethod
    def get_params_path_from_settings() -> str:
        run = ModelRun.get_run_from_settings()
        params_path = PathConstructor.get_params_path(run)
        return params_path

    @staticmethod
    def get_model_path_from_settings() -> str:
        run = ModelRun.get_run_from_settings()
        model_path = PathConstructor.get_model_path(run)
        return model_path

    @staticmethod
    def get_metrics_path_from_settings() -> str:
        run = ModelRun.get_run_from_settings()
        metrics_path = PathConstructor.get_params_path(run)
        return metrics_path
