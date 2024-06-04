import yaml


class FileReader:
    def __init__(self, path):
        self.path = path

    def set_path(self, path):
        self.path = path

    def get_path(self):
        return self.path

    def read_yaml(self):
        with open(self.path, 'r') as file:
            data = yaml.safe_load(file)
        return data

    def read_sql(self):
        with open(self.path, 'r') as file:
            sql = file.read()
            file.close()
            return sql
