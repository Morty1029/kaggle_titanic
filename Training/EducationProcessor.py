from Training.ModelTypes import ModelTypes
from ConfigsClasses.Settings import Settings
from Dataset_classes.DatasetSplitter import DatasetSplitter, SplitDataset


class EducationProcessor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = None
        self.model_type = None
        self.split_data = SplitDataset()
        self.__set_model()

    def train_model(self, get_val=True):
        splitter = DatasetSplitter(self.dataset)
        self.split_data = splitter.split_dataset(get_val)
        self.model.fit(self.split_data.x_train, self.split_data.y_train, **(self.__get_additional_params()))

    def get_results(self):
        predictions = self.model.predict(self.split_data.x_test)
        return predictions

    def __set_model(self):
        settings = Settings()
        self.model_type = ModelTypes[settings.model_type]
        self.model = self.model_type.value

    def __get_additional_params(self):
        if self.model_type == ModelTypes.CATBOOST:
            return {
                'eval_set': (self.split_data.x_test, self.split_data.y_test),
                'early_stopping_rounds': 10,
                'cat_features': self.dataset.cat_cols
            }
        else:
            # TODO
            return {}
