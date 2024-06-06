from sklearn.model_selection import train_test_split
import random
from Training.ModelTypes import ModelTypes
from ConfigsClasses.Settings import Settings


class EducationProcessor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = None
        self.model_type = None
        self.x_train = None
        self.x_val = None
        self.x_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.__set_model()

    def train_model(self, get_val=True):
        if not get_val:
            self.x_train, self.x_test, self.y_train, self.y_test = self.get_test_train(get_val)
        else:
            self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = self.get_test_train(get_val)
        self.model.fit(self.x_train, self.y_train, **(self.__get_additional_params()))

    def get_test_train(self, get_val=True):
        data = self.dataset.df
        target_cols = self.dataset.target_cols
        x = data.drop(columns=self.dataset.date_cols + target_cols, axis=1)
        y = data[target_cols]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                            random_state=int(random.random() * 1000), shuffle=True)
        if not get_val:
            return x_train, x_test, y_train, y_test

        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=True)
        return x_train, x_val, x_test, y_train, y_val, y_test

    def get_results(self):
        predictions = self.model.predict(self.x_test)
        return predictions

    def __set_model(self):
        settings = Settings()
        self.model_type = ModelTypes[settings.model_type]
        self.model = self.model_type.value

    def __get_additional_params(self):
        if self.model_type == ModelTypes.CATBOOST:
            return {
                'eval_set': (self.x_test, self.y_test),
                'early_stopping_rounds': 10,
                'cat_features': self.dataset.cat_cols
            }
        else:
            # TODO
            return {}
