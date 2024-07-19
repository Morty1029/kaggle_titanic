from dataset_classes.Dataset import Dataset
import pandas as pd


class DatasetFilter:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def clear(self):
        self.delete_useless_cols()
        self.feel_missing_values_default()
        self.bool_cols_into_int()

    def bool_cols_into_int(self):
        if self.dataset.df is not None and isinstance(self.dataset.df, pd.DataFrame):
            for col in self.dataset.bool_cols:
                self.dataset.df[col] = self.dataset.df[col].astype(int)
        else:
            print('no_data')
            return
        print('bool values were converted into int')

    def feel_missing_values_default(self):
        for col in self.dataset.bool_cols + self.dataset.num_cols:
            self.dataset.df[col] = self.dataset.df[col].fillna(0)
        for col in self.dataset.cat_cols:
            self.dataset.df[col] = self.dataset.df[col].fillna('skipped')
        print('missing values were fel by default')

    def delete_useless_cols(self):
        self.dataset.df = self.dataset.df.drop(self.dataset.useless_cols, axis=1)
        print('useless cols were deleted')
