from decorators.patterns import singleton
from utils.FileReader import FileReader
from enums.Paths import Paths
from dataclasses import dataclass
from training.model_utils.ModelTypes import ModelTypes
from enums.ModelStages import ModelStages
from enums.TaskTypes import TaskTypes


@singleton
@dataclass
class Settings:
    def __init__(self):
        settings = FileReader.read_yaml(Paths.SETTINGS_PATH.value)
        self.get_report = settings['get_report']
        self.draw_origin_graphs = settings['draw_origin_graphs']
        self.model_type = ModelTypes[settings['model_type']]
        self.version = settings['version']
        self.stage = ModelStages[settings['stage']]
        self.experiment_name = settings['experiment_name']
        self.task_type = TaskTypes(settings['task_type'])
