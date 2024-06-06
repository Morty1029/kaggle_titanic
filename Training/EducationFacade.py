from Training.EducationProcessor import EducationProcessor
from Utils.Metrics.MetricsFacade import MetricsFacade
from Training.ModelsLogger import ModelsLogger
from Training.ModelRun import ModelRun
from ConfigsClasses.Settings import Settings
from Utils.ModelPathConstructor import ModelPathConstructor


class EducationFacade:
    def __init__(self, dataset, get_val, classification=True):
        self.dataset = dataset
        self.get_val = get_val
        self.classification = classification

    def get_results(self):
        processor = EducationProcessor(self.dataset)
        processor.train_model(self.get_val)
        mpc = ModelPathConstructor()
        settings = Settings()
        run = ModelRun(model=processor.model,
                       model_type=processor.model_type,
                       name=processor.model_type.name,
                       version=settings.version,
                       stage=settings.stage)
        mpc.set_model_run(run)
        predictions = processor.get_results()
        metrics = MetricsFacade(predictions, processor.split_data.y_test, path=mpc.get_metrics_path())
        metrics.give_me_metrics()
        logger = ModelsLogger(model=processor.model,
                              model_type=processor.model_type)
        logger.model_to_file(mpc.get_model_path())
