from Utils.Graphs.Graph import Graph
import matplotlib.pyplot as plt


class GraphDrawer:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.__set_plt_properties()

    def __set_plt_properties(self):
        plt.figure(figsize=self.graph.size)
        plt.title(self.graph.title)
        plt.xlabel(self.graph.x_label)
        plt.ylabel(self.graph.y_label)

    def draw_graph(self):
        if self.graph.show:
            plt.show()
