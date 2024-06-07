import pandas as pd
from ydata_profiling import ProfileReport
import webbrowser
from Utils.FileReader import FileReader


class Dataset:
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.num_cols = []
        self.cat_cols = []
        self.useless_cols = []
        self.target_cols = []
        self.bool_cols = []
        self.date_cols = []
        self.path_to_report = ''
        self.report = None
        print('Dataset was created')

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

    def clear(self):
        self.delete_useless_cols()
        self.bool_cols_into_int()
        self.feel_missing_values_default()

    def bool_cols_into_int(self):
        if self.df is not None and isinstance(self.df, pd.DataFrame):
            for col in self.bool_cols:
                self.bool_col_into_int(col)
        else:
            print('no_data')
            return
        print('bool values were converted into 1 0')

    def bool_col_into_int(self, col):
        if self.df is not None and isinstance(self.df, pd.DataFrame):
            if col is not None:
                self.df[col] = self.df[col].apply(lambda x: 1 if x else 0)
            else:
                print('no col')
        else:
            print('no data')

    def int_col_into_bool(self, col):
        if self.df is not None and isinstance(self.df, pd.DataFrame):
            if col is not None:
                self.df[col] = self.df[col].apply(lambda x: x == 1)
            else:
                print('no col')
        else:
            print('no data')

    def feel_missing_values_default(self):
        for col in self.bool_cols + self.num_cols:
            self.df[col] = self.df[col].fillna(0)
        for col in self.cat_cols:
            self.df[col] = self.df[col].fillna('skipped')
        print('missing values were fel by default')

    def delete_useless_cols(self):
        self.df = self.df.drop(self.useless_cols, axis=1)
