import json
import mlflow


class DictLogger:
    @staticmethod
    def dict_to_json(some_dict: dict, path: str):
        with open(path + '.json', 'w+') as file:
            json.dump(some_dict, file)
