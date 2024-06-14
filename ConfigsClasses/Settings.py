from Decorators.patterns import singleton
from Utils.FileReader import FileReader
from Enums.Paths import Paths
from dataclasses import dataclass


@singleton
@dataclass
class Settings:
    def __init__(self):
        settings = FileReader.read_yaml(Paths.SETTINGS_PATH.value)
        self.get_report = settings['get_report']
        self.draw_origin_graphs = settings['draw_origin_graphs']
        self.model_type = settings['model_type']
        self.version = settings['version']
        self.stage = settings['stage']
        self.experiment_name = settings['experiment_name']
