from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, precision_score, \
    recall_score, mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error

from math import pow


class MetricsCounter:
    def __init__(self, y_test, predictions, round_to=4):
        self.y_test = y_test
        self.predictions = predictions
        self.classification_metrics = {}
        self.regression_metrics = {}
        self.round_to = round_to

    def set_classification_metrics(self):
        self.classification_metrics = {
            'accuracy': round(accuracy_score(self.y_test, self.predictions), self.round_to),
            'roc_auc': round(roc_auc_score(self.y_test, self.predictions), self.round_to),
            'precision': round(precision_score(self.y_test, self.predictions), self.round_to),
            'recall': round(recall_score(self.y_test, self.predictions), self.round_to),
            'cohen kappa': round(cohen_kappa_score(self.y_test, self.predictions), self.round_to)
        }

    def get_classification_metrics(self):
        return self.classification_metrics

    def set_regression_metrics(self):
        mse = mean_squared_error(self.y_test, self.predictions)
        self.regression_metrics = {
            'RMSE': round(pow(mse, 0.5), self.round_to),
            'r2': round(r2_score(self.y_test, self.predictions), self.round_to),
            'MAE': round(mean_absolute_error(self.y_test, self.predictions), self.round_to),
            'MAPE': round(mean_absolute_percentage_error(self.y_test, self.predictions), self.round_to)
        }

    def get_regression_metrics(self):
        return self.regression_metrics



