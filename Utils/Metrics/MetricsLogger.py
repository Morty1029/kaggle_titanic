import json


class MetricsLogger:
    def __init__(self, metrics: dict):
        self.metrics = metrics

    def metrics_to_json(self, path):
        with open(path + '.json', 'w+') as file:
            json.dump(self.metrics, file)
