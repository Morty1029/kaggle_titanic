from dataclasses import dataclass
from training.model_utils.ModelTypes import ModelTypes


@dataclass
class ModelObject:
    def __init__(self, model, model_type: ModelTypes):
        self.model = model
        self.model_type = model_type
