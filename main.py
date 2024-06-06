from Training.EducationFacade import EducationFacade
from Enums.Paths import Paths
from Dataset_classes.Dataset import Dataset
from Dataset_classes.DatasetVisualizer import DatasetVisualizer
from ConfigsClasses.Settings import Settings


SETTINGS = Settings()
TARGET_COL = 'Transported'


def main():
    dataset = Dataset(Paths.DATASET_PATH.value)
    if SETTINGS.get_report:
        dataset.get_new_report('res\\report.html')
    get_baseline(dataset)


def get_baseline(dataset: Dataset):
    dataset.set_cols_from(Paths.BASELINE_DATA_CONFIG.value)
    dataset.clear()
    if SETTINGS.draw_origin_graphs:
        draw_origin_graph(dataset)
    education_facade = EducationFacade(dataset, False)
    education_facade.get_results()


def draw_origin_graph(dataset):
    visualizer = DatasetVisualizer(dataset)
    visualizer.draw_pair_counts(TARGET_COL)
    visualizer.draw_pair_hists(TARGET_COL)


if __name__ == "__main__":
    main()
