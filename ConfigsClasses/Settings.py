from Decorators.patterns import singleton
from Utils.FileReader import FileReader
from Enums.Paths import Paths
from dataclasses import dataclass


@singleton
class Settings:
    def __init__(self):
        reader = FileReader()
        reader.set_path(Paths.SETTINGS_PATH.value)
        settings = reader.read_yaml()
        self.get_report = settings['get_report']
        self.draw_origin_graphs = settings['draw_origin_graphs']
        self.model_type = settings['model_type']
        self.version = settings['version']
        self.stage = settings['stage']
