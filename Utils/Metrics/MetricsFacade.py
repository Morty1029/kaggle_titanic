from Utils.Metrics.MetricsLogger import MetricsLogger
from Utils.Metrics.MetricsPrinter import MetricsPrinter
from Utils.Metrics.MetricsCounter import MetricsCounter


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

    def give_me_metrics(self):
        counter = MetricsCounter(self.y_test, self.predictions, self.round_to)
        if self.classification:
            counter.set_classification_metrics()
            metrics = counter.get_classification_metrics()
        else:
            counter.set_regression_metrics()
            metrics = counter.get_regression_metrics()
        printer = MetricsPrinter(metrics)
        logger = MetricsLogger(metrics)
        printer.print_metrics()
        logger.metrics_to_json(self.path)
