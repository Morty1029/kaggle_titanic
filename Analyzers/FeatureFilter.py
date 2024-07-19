from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from preprocessing.Preprocessor import Preprocessor
import numpy as np
from config_classes.Settings import Settings
from enums.TaskTypes import TaskTypes
from scipy.stats import ttest_ind


class FeatureFilter:
    def __init__(self, dataset, metric: str = 'accuracy'):
        self.dataset = dataset.__copy__()
        self._name_of_useless_cols = 'useless_col_'
        self.num_of_useless_cols = 1
        self.settings = Settings()
        self.metric = metric

    def get_useless_cols(self) -> list[str]:
        self.__add_noise_cols(self.num_of_useless_cols)
        importance_cols = self.importance_useless_cols()
        cv_cols = self.cv_useless_cols(importance_cols)
        return list(set(importance_cols) & set(cv_cols))

    def cv_useless_cols(self, cols=None) -> list:
        forest = self.__get_forest()
        useless_cols = []
        alpha = 0.95
        for col in cols:
            x, y = self.dataset.get_x_y()
            with_col = cross_val_score(forest, x, y, scoring=self.metric, cv=10)
            df = self.dataset.__copy__()
            x, y = df.get_x_y()
            without_col = cross_val_score(forest, x, y, scoring=self.metric, cv=10)
            t_stat, p_value = ttest_ind(with_col, without_col)
            if alpha <= p_value:
                useless_cols.append(col)
            del x, y, df
        return useless_cols

    def importance_useless_cols(self) -> list[str]:
        lasso_dict = self.__lasso_get_cols_coef_dict()
        forest_dict = self.__forest_get_cols_coef_dict()
        lr_cols = self.__get_useless_cols_by_dict(lasso_dict)
        forest_cols = self.__get_useless_cols_by_dict(forest_dict)
        cols = list(set(lr_cols) ^ set(forest_cols))
        return cols

    def __get_useless_cols_by_dict(self, cols_coef_dict) -> list:
        useless_coef = self.__get_useless_coef(cols_coef_dict)
        threshold = sum(useless_coef) / len(useless_coef)
        noisy_cols = self.__get_names_noisy_cols()
        return [k for k, v in cols_coef_dict.items() if abs(v) <= threshold and k not in noisy_cols]

    def __lasso_get_cols_coef_dict(self) -> dict:
        log_reg = self.__get_lasso()
        data = self.dataset.__copy__()
        preprocessor = Preprocessor(data)
        preprocessor.ohe()
        preprocessor.standard_scaler()
        x, y = data.get_x_y()
        log_reg.fit(x, y)
        cols_coef_dict = dict(zip(log_reg.feature_names_in_, log_reg.coef_[0].tolist()))
        return cols_coef_dict

    def __forest_get_cols_coef_dict(self) -> dict:
        forest = self.__get_forest()
        preprocessor = Preprocessor(dataset=self.dataset)
        preprocessor.ohe()
        x, y = self.dataset.get_x_y()
        forest.fit(x, y)
        cols_coef_dict = dict(zip(forest.feature_names_in_, forest.feature_importances_.tolist()))
        return cols_coef_dict

    def __get_names_noisy_cols(self) -> list[str]:
        cols = []
        for i in range(self.num_of_useless_cols):
            cols.append(self._name_of_useless_cols + str(i))
        return cols

    def __get_useless_coef(self, cols_coef_dict: dict) -> list:
        coef = []
        for i in range(self.num_of_useless_cols):
            useless_col_name = self._name_of_useless_cols + str(i)
            i_coef = abs(cols_coef_dict.get(useless_col_name))
            coef.append(i_coef)
        return coef

    def __add_noise_cols(self, num=1):
        for i in range(num):
            self.__add_noise_col(self._name_of_useless_cols + str(i))

    def __add_noise_col(self, col_name: str):
        self.dataset.df[col_name] = np.random.rand(len(self.dataset.df))

    def __get_lasso(self):
        if self.settings.task_type == TaskTypes.CLASSIFICATION:
            log_reg = LogisticRegression(penalty='l1', solver='saga', max_iter=10000, C=10)
        else:
            log_reg = Lasso()
        return log_reg

    def __get_forest(self):
        if self.settings.task_type == TaskTypes.CLASSIFICATION:
            forest = RandomForestClassifier()
        else:
            forest = RandomForestRegressor()
        return forest
