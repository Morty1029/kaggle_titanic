import json


class DictLogger:
    @staticmethod
    def dict_to_json(some_dict, path):
        with open(path + '.json', 'w+') as file:
            json.dump(some_dict, file)
