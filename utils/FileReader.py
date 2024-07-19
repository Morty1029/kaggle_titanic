import yaml
import json


class FileReader:

    @staticmethod
    def read_yaml(path: str) -> dict:
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
        return data

    @staticmethod
    def read_sql(path: str) -> str:
        with open(path, 'r') as file:
            sql = file.read()
            file.close()
            return sql

    @staticmethod
    def read_json(path: str) -> dict:
        with open(path, 'r') as file:
            some_json = json.load(file)
        return some_json
