from dataclasses import dataclass


@dataclass
class ModelRun:
    def __init__(self, model, name, version, stage):
        self.model = model
        self.name = name
        self.version = version
        self.stage = stage

    def __str__(self):
        return self.name + ' v' + self.version + ' ' + self.stage
