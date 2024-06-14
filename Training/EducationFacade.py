import mlflow

from Training.EducationProcessor import EducationProcessor
from Utils.Metrics.MetricsFacade import MetricsFacade
from Training.ModelsLogger import ModelsLogger
from Training.ModelRun import ModelRun
from ConfigsClasses.Settings import Settings
from Utils.ModelPathConstructor import ModelPathConstructor
from Training.ModelConstructor import ModelConstructor
from typing import Callable
from Training.ModelOptimizer import ModelOptimizer


class EducationFacade:
    def __init__(self, dataset, get_val, metric: Callable, classification=True):
        self.dataset = dataset
        self.get_val = get_val
        self.classification = classification
        self.metric = metric

    def get_results(self):
        settings = Settings()
        mlflow.set_tracking_uri('http://localhost:5000')
        mlflow.set_experiment(settings.experiment_name)
        with mlflow.start_run():
            model_object = ModelConstructor.get_model(settings.model_type)
            run = ModelRun(model_object=model_object,
                           name=model_object.model_type.name,
                           version=settings.version,
                           stage=settings.stage)
            optimizer = ModelOptimizer(model_object=model_object,
                                       dataset=self.dataset,
                                       metric=self.metric,
                                       path_to_best_params=ModelPathConstructor.get_params_path(run))
            optimizer.optimize_model()
            processor = EducationProcessor(self.dataset, model_object, optimizer.best_params)
            processor.train_model(self.get_val)
            predictions = processor.get_results()
            metrics_facade = MetricsFacade(predictions=predictions,
                                           y_test=processor.split_data.y_test,
                                           path=ModelPathConstructor.get_metrics_path(run))
            metrics_facade.give_me_metrics()
            ModelsLogger.model_to_file(model_object=model_object,
                                       path=ModelPathConstructor.get_model_path(run))
            mlflow.log_params(processor.params)
            ModelsLogger.model_to_mlflow(model_run=run)
