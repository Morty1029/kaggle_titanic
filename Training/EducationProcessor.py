from Training.ModelTypes import ModelTypes
from ConfigsClasses.Settings import Settings
from Dataset_classes.DatasetSplitter import DatasetSplitter, SplitDataset
from Training.ModelObject import ModelObject
from Dataset_classes.Dataset import Dataset
from Training.ModelConstructor import ModelConstructor


class EducationProcessor:
    def __init__(self, dataset: Dataset, model_object: ModelObject, params=None):
        self.dataset = dataset
        self.split_data = SplitDataset()
        self.model_object = model_object
        self.params = params

    def train_model(self, get_val=True):
        splitter = DatasetSplitter(self.dataset)
        self.split_data = splitter.split_dataset(get_val)
        self.model_object.model.set_params(**self.params)
        additional_params = ModelConstructor.get_additional_params(self.model_object.model_type,
                                                                   self.split_data,
                                                                   self.dataset)
        self.model_object.model.fit(self.split_data.x_train,
                                    self.split_data.y_train,
                                    **additional_params)

    def get_results(self):
        predictions = self.model_object.model.predict(self.split_data.x_test)
        return predictions
