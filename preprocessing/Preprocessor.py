from dataset_classes.Dataset import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Preprocessor:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def ohe(self):
        df = self.dataset.df
        cols = self.dataset.cat_cols
        self.dataset.df = pd.get_dummies(df, columns=cols)

    def standard_scaler(self):
        scaler = StandardScaler()
        self.__scale(scaler)

    def min_max_scaler(self):
        scaler = MinMaxScaler()
        self.__scale(scaler)

    def __scale(self, scaler):
        for col in self.dataset.num_cols:
            self.dataset.df[col] = scaler.fit_transform(self.dataset.df[col].to_numpy().reshape((-1, 1)))
