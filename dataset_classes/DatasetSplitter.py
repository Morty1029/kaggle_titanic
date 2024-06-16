from dataset_classes.Dataset import Dataset
from sklearn.model_selection import train_test_split
import random
from dataset_classes.SplitDataset import SplitDataset


class DatasetSplitter:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def split_dataset(self, get_val=True) -> SplitDataset:
        data = self.dataset.df
        target_cols = self.dataset.target_cols
        x = data.drop(columns=self.dataset.date_cols + target_cols, axis=1)
        y = data[target_cols]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                            random_state=int(random.random() * 1000), shuffle=True)
        if not get_val:
            return SplitDataset(x_train=x_train,
                                x_test=x_test,
                                y_train=y_train,
                                y_test=y_test)

        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=True)
        return SplitDataset(x_train=x_train,
                            x_val=x_val,
                            x_test=x_test,
                            y_train=y_train,
                            y_val=y_val,
                            y_test=y_test)
