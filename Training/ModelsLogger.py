from Training.ModelTypes import ModelTypes


class ModelsLogger:
    def __init__(self, model, model_type: ModelTypes):
        self.model = model
        self.model_type = model_type

    def model_to_file(self, path):
        if self.model_type == ModelTypes.CATBOOST:
            self.model.save_model(path, format='cbm')
        pass
