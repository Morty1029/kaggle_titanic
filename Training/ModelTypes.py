from enum import Enum
from catboost import CatBoostClassifier


class ModelTypes(Enum):
    CATBOOST = CatBoostClassifier(thread_count=30)
