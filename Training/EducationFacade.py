from Training.EducationProcessor import EducationProcessor
from Utils.Metrics.MetricsFacade import MetricsFacade


class EducationFacade:
    def __init__(self, dataset, get_val, classification=True):
        self.dataset = dataset
        self.get_val = get_val
        self.classification = classification

    def get_results(self):
        processor = EducationProcessor(self.dataset)
        processor.train_model(self.get_val)
        predictions = processor.get_results()
        metrics = MetricsFacade(predictions, processor.x_test)
        metrics.give_me_metrics()
