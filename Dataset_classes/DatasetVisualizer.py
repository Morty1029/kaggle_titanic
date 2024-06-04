import seaborn as sns
from Dataset_classes.Dataset import Dataset
from Utils.Graphs.GraphFacade import GraphFacade
from math import log2


class DatasetVisualizer:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def draw_pair_counts(self, target_col):
        for col in self.dataset.cat_cols + self.dataset.bool_cols:
            sns.countplot(self.dataset.df, x=col, hue=target_col)
            drawer = GraphFacade((14, 8), f'{col}/{target_col}', col, 'Count')
            drawer.draw_graph()

    def draw_pair_hists(self, target_col):
        for col in self.dataset.num_cols:
            bins = self.__get_number_of_bins(col)
            sns.histplot(data=self.dataset.df, x=col, hue=target_col, multiple='stack', bins=bins, kde=True)
            drawer = GraphFacade((14, 8), f'{col}/{target_col}', col, 'Freq')
            drawer.draw_graph()

    def __get_number_of_bins(self, col):
        df_col = self.dataset.df[col]
        n = df_col.max() - df_col.min()
        return round(1 + log2(n), 0)
