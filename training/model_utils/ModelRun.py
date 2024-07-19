from dataclasses import dataclass
from training.model_utils.ModelObject import ModelObject
from enums.ModelStages import ModelStages
from config_classes.Settings import Settings
from training.model_utils.ModelTypes import ModelTypes


@dataclass
class ModelRun:
    def __init__(self, model_object: ModelObject, name, version, stage: ModelStages):
        self.model_object = model_object
        self.name = name
        self.version = version
        self.stage = stage

    def __str__(self):
        return f'{self.name}({self.model_object.model_type.name})_v{self.version}_{self.stage.name}'

    @staticmethod
    def get_run_from_settings():
        settings = Settings()
        model_type = settings.model_type
        model_object = ModelObject(model=model_type.value, model_type=model_type)
        return ModelRun(model_object=model_object,
                        name=settings.experiment_name,
                        version=settings.version,
                        stage=settings.stage)
