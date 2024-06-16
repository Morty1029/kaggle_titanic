from utils.DictLogger import DictLogger
from utils.metrics.MetricsPrinter import MetricsPrinter
from utils.metrics.MetricsCounter import MetricsCounter
import mlflow
from config_classes.Settings import Settings
from enums.TaskTypes import TaskTypes


class MetricsFacade:
    def __init__(self, predictions,
                 y_test,
                 round_to=4,
                 path='test_metrics'):
        self.predictions = predictions
        self.y_test = y_test
        self.round_to = round_to
        self.path = path
        self.metrics = {}

    def give_me_metrics(self):
        settings = Settings()
        counter = MetricsCounter(self.y_test, self.predictions, self.round_to)
        if settings.task_type == TaskTypes.CLASSIFICATION:
            counter.set_classification_metrics()
            self.metrics = counter.get_classification_metrics()
        else:
            counter.set_regression_metrics()
            self.metrics = counter.get_regression_metrics()
        MetricsPrinter.print_metrics(self.metrics)
        DictLogger.dict_to_json(self.metrics, self.path)
        mlflow.log_metrics(self.metrics)
