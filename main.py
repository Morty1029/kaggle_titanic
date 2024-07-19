from training.education.EducationFacade import EducationFacade
from enums.Paths import Paths
from dataset_classes.Dataset import Dataset
from dataset_classes.DatasetVisualizer import DatasetVisualizer
from config_classes.Settings import Settings
from sklearn.metrics import accuracy_score
from enums.ModelStages import ModelStages
from dataset_classes.DatasetFilter import DatasetFilter

SETTINGS = Settings()
TARGET_COL = 'Transported'


def main():
    dataset = Dataset()
    dataset.load_data_from(Paths.DATASET_PATH.value)
    if SETTINGS.get_report:
        dataset.get_new_report('res\\report.html')
    get_baseline(dataset)


def get_baseline(dataset: Dataset):
    dataset.set_cols_from(Paths.BASELINE_DATA_CONFIG.value)
    data_filter = DatasetFilter(dataset)
    data_filter.clear()
    if SETTINGS.draw_origin_graphs:
        draw_origin_graph(dataset)
    education_facade = EducationFacade(dataset)
    if SETTINGS.stage == ModelStages.STAGING:
        education_facade.start_experiment(get_val=False, metric=accuracy_score)
    elif SETTINGS.stage == ModelStages.PRODUCTION:
        path_to_params = "results\\CATBOOST\\params\\params_baseline(CATBOOST)_v1_STAGING.json"
        education_facade.train_saved_model(path_to_params=path_to_params)


def draw_origin_graph(dataset):
    visualizer = DatasetVisualizer(dataset)
    visualizer.draw_pair_counts(TARGET_COL)
    visualizer.draw_pair_hists(TARGET_COL)


if __name__ == "__main__":
    main()
