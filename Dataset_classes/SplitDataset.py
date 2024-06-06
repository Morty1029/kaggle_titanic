from dataclasses import dataclass


@dataclass
class SplitDataset:
    def __init__(self,
                 x_train=None,
                 x_val=None,
                 x_test=None,
                 y_train=None,
                 y_val=None,
                 y_test=None):
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
