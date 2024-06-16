from enum import Enum


class ModelStages(Enum):
    STAGING = 'staging'
    PRODUCTION = 'production'
    ARCHIVED = 'archived'
