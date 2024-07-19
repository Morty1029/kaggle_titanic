import pandas as pd
from ydata_profiling import ProfileReport
import webbrowser
from utils.FileReader import FileReader
from typing import Union


class Dataset:
    def __init__(self):
        self.df: Union[pd.DataFrame, None] = None
        self.num_cols = []
        self.cat_cols = []
        self.useless_cols = []
        self.target_cols = []
        self.bool_cols = []
        self.date_cols = []
        self.path_to_report = ''
        self.report = None

    def __copy__(self):
        df = None if self.df is None else self.df.copy()
        dataset = Dataset()
        dataset.df = df
        dataset.num_cols = self.num_cols.copy()
        dataset.cat_cols = self.cat_cols.copy()
        dataset.useless_cols = self.useless_cols.copy()
        dataset.target_cols = self.target_cols.copy()
        dataset.bool_cols = self.bool_cols.copy()
        dataset.date_cols = self.date_cols.copy()
        return dataset

    def load_data_from(self, path: str):
        self.df = pd.read_csv(path)

    def set_data(self, df: pd.DataFrame):
        self.df = df

    def set_cols_from(self, path_to_config):
        config = FileReader.read_yaml(path_to_config)
        self.num_cols = config['num_cols']
        self.cat_cols = config['cat_cols']
        self.target_cols = config['target_cols']
        self.useless_cols = config['useless_cols']
        self.bool_cols = config['bool_cols']
        self.date_cols = config['date_cols']
        print('Cols are set')

    def get_new_report(self, path_to_report):
        self.report_to_file(path_to_report)
        self.get_report()

    def report_to_file(self, path_to_report):
        self.report = ProfileReport(self.df, minimal=True, title='Report')
        self.report.to_file(path_to_report)
        self.path_to_report = path_to_report
        print('report was saved')

    def get_report(self):
        if self.df is not None:
            if self.report is not None:
                if self.path_to_report != '':
                    webbrowser.open(self.path_to_report)
                    print('report were opened')
                else:
                    print('no path to report')
            else:
                print('no report')
        else:
            print('no data - no report')

    def get_x_y(self):
        x = self.df.drop(self.target_cols + self.date_cols, axis=1)
        y = self.df[self.target_cols]
        return x, y
