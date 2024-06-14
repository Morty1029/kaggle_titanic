from Utils.DictLogger import DictLogger
from Utils.Metrics.MetricsPrinter import MetricsPrinter
from Utils.Metrics.MetricsCounter import MetricsCounter
import mlflow

class MetricsFacade:
    def __init__(self, predictions,
                 y_test,
                 round_to=4,
                 classification=True,
                 path='test_metrics'):
        self.predictions = predictions
        self.y_test = y_test
        self.round_to = round_to
        self.classification = classification
        self.path = path
        self.metrics = {}

    def give_me_metrics(self):
        counter = MetricsCounter(self.y_test, self.predictions, self.round_to)
        if self.classification:
            counter.set_classification_metrics()
            self.metrics = counter.get_classification_metrics()
        else:
            counter.set_regression_metrics()
            self.metrics = counter.get_regression_metrics()
        MetricsPrinter.print_metrics(self.metrics)
        DictLogger.dict_to_json(self.metrics, self.path)
        mlflow.log_metrics(self.metrics)

