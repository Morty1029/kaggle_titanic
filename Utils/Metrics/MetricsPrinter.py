class MetricsPrinter:
    def __init__(self, metrics: dict):
        self.metrics = metrics

    def print_metrics(self):
        for metric, value in self.metrics.items():
            print(f'{metric} = {value}\n')
