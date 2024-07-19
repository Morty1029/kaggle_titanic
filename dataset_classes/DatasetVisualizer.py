import seaborn as sns
from dataset_classes.Dataset import Dataset
from math import log2
import matplotlib.pyplot as plt


class DatasetVisualizer:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def draw_pair_counts(self, target_col):
        for col in self.dataset.cat_cols + self.dataset.bool_cols:
            sns.countplot(data=self.dataset.df, x=col, hue=target_col)
            plt.show()

    def draw_pair_hists(self, target_col):
        for col in self.dataset.num_cols:
            bins = self.__get_number_of_bins(col)
            sns.histplot(data=self.dataset.df, x=col, hue=target_col, multiple='stack', bins=bins, kde=True)
            plt.show()

    def draw_heatmap(self):
        sns.heatmap(self.dataset.df.corr(), annot=False, fmt=".3f")
        plt.show()

    def __get_number_of_bins(self, col):
        df_col = self.dataset.df[col]
        n = df_col.max() - df_col.min()
        return int(1 + log2(n))
