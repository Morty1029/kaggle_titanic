import mlflow
from training.education.EducationProcessor import EducationProcessor
from utils.metrics.MetricsFacade import MetricsFacade
from training.model_utils.ModelsLogger import ModelsLogger
from training.model_utils.ModelRun import ModelRun
from config_classes.Settings import Settings
from utils.PathConstructor import PathConstructor
from training.model_utils.ModelConstructor import ModelConstructor
from typing import Callable
from training.model_utils.ModelOptimizer import ModelOptimizer


class EducationFacade:
    def __init__(self, dataset, get_val, metric: Callable):
        self.dataset = dataset
        self.get_val = get_val
        self.metric = metric

    def train_model(self):
        settings = Settings()
        mlflow.set_tracking_uri('http://localhost:5000')
        mlflow.set_experiment(settings.experiment_name)
        with mlflow.start_run():
            model_object = ModelConstructor.get_model_of_type(settings.model_type)
            run = ModelRun(model_object=model_object,
                           name=settings.experiment_name,
                           version=settings.version,
                           stage=settings.stage)
            mlflow.set_tag('mlflow.runName', run.__str__())
            optimizer = ModelOptimizer(model_object=model_object,
                                       dataset=self.dataset,
                                       metric=self.metric,
                                       path_to_best_params=PathConstructor.get_params_path(run))
            optimizer.optimize_model()
            processor = EducationProcessor(self.dataset, model_object, optimizer.best_params)
            processor.train_model(self.get_val)
            predictions = processor.get_results()
            metrics_facade = MetricsFacade(predictions=predictions,
                                           y_test=processor.split_data.y_test,
                                           path=PathConstructor.get_metrics_path(run))
            metrics_facade.give_me_metrics()
            ModelsLogger.model_to_file(model_object=model_object,
                                       path=PathConstructor.get_model_path(run))
            mlflow.log_params(processor.params)
            ModelsLogger.model_to_mlflow(model_run=run)
