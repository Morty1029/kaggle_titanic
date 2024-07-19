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
from utils.FileReader import FileReader


class EducationFacade:
    def __init__(self, dataset):
        self.dataset = dataset

    def start_experiment(self, get_val, metric: Callable):
        settings = Settings()
        mlflow.set_tracking_uri('http://localhost:5000')
        mlflow.set_experiment(settings.experiment_name)
        with mlflow.start_run():
            run = ModelRun.get_run_from_settings()
            model_object = run.model_object
            mlflow.set_tag('mlflow.runName', run.__str__())
            optimizer = ModelOptimizer(model_object=model_object,
                                       dataset=self.dataset,
                                       metric=metric,
                                       path_to_best_params=PathConstructor.get_params_path(run))
            optimizer.optimize_model()
            processor = EducationProcessor(self.dataset, model_object, optimizer.best_params)
            processor.train_model(get_val)
            predictions = processor.get_test_results()
            metrics_facade = MetricsFacade(predictions=predictions,
                                           y_test=processor.split_data.y_test,
                                           path=PathConstructor.get_metrics_path(run))
            metrics_facade.give_me_metrics()
            ModelsLogger.model_to_file(model_object=model_object,
                                       path=PathConstructor.get_model_path(run))
            mlflow.log_params(processor.params)
            ModelsLogger.model_to_mlflow(model_run=run)

    def train_saved_model(self, path_to_params):
        model = ModelConstructor.get_model_of_type()
        params = FileReader.read_json(path_to_params)
        processor = EducationProcessor(dataset=self.dataset,
                                       model_object=model,
                                       params=params,
                                       use_split=False)
        processor.train_model()
        ModelsLogger.model_to_file(model_object=model, path=PathConstructor.get_model_path_from_settings())
