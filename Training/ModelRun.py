from dataclasses import dataclass
from Training.ModelTypes import ModelTypes
from Enums.ModelStages import ModelStages


@dataclass
class ModelRun:
    def __init__(self, model, model_type: ModelTypes, name, version, stage: ModelStages):
        self.model = model
        self.model_type = model_type
        self.name = name
        self.version = version
        self.stage = stage

    def __str__(self):
        return f'{self.name}({self.model_type.name})_v{self.version}_{self.stage}'
