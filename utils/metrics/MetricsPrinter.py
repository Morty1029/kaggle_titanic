class MetricsPrinter:

    @staticmethod
    def print_metrics(metrics_dict: dict):
        for metric, value in metrics_dict.items():
            print(f'{metric} = {value}\n')
