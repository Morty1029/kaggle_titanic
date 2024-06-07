from dataclasses import dataclass
from Training.ModelObject import ModelObject
from Enums.ModelStages import ModelStages


@dataclass
class ModelRun:
    def __init__(self, model_object: ModelObject, name, version, stage: ModelStages):
        self.model_object = model_object
        self.name = name
        self.version = version
        self.stage = stage

    def __str__(self):
        return f'{self.name}({self.model_object.model_type.name})_v{self.version}_{self.stage}'
