from dataclasses import dataclass


@dataclass
class ModelObject:
    def __init__(self, model, model_type):
        self.model = model
        self.model_type = model_type
