from training.model_utils.ModelConstructor import ModelConstructor, ModelTypes


model = ModelConstructor.get_model_from_file('results\\CATBOOST\\models\\model_baseline(CATBOOST)_v1_STAGING', ModelTypes.CATBOOST)
a = 1