from dataset_classes.DatasetSplitter import DatasetSplitter, SplitDataset
from training.model_utils.ModelObject import ModelObject
from dataset_classes.Dataset import Dataset
from training.model_utils.ModelConstructor import ModelConstructor


class EducationProcessor:
    def __init__(self, dataset: Dataset, model_object: ModelObject, params=None, use_split: bool = True):
        self.dataset = dataset
        self.split_data = SplitDataset()
        self.model_object = model_object
        self.params = params
        self.use_split = use_split

    def train_model(self, get_val=True):
        self.model_object.model.set_params(**self.params)
        additional_params = ModelConstructor.get_additional_params(self.model_object.model_type,
                                                                   dataset=self.dataset)
        if self.use_split:
            splitter = DatasetSplitter(self.dataset)
            self.split_data = splitter.split_dataset(get_val)
            self.model_object.model.fit(self.split_data.x_train,
                                        self.split_data.y_train,
                                        **additional_params)
        else:
            x = self.dataset.df.drop(self.dataset.date_cols + self.dataset.target_cols, axis=1)
            y = self.dataset.df[self.dataset.target_cols]
            self.model_object.model.fit(x, y, **additional_params)

    def get_test_results(self):
        predictions = self.model_object.model.predict(self.split_data.x_test)
        return predictions
