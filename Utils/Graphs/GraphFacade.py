from Utils.Graphs.GraphDrawer import Graph, GraphDrawer


class GraphFacade:
    def __init__(self, size, title, x_label, y_label, show=True):
        self.size = size
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.show = show

    def draw_graph(self):
        graph = Graph(self.size, self.title, self.x_label, self.y_label, self.show)
        drawer = GraphDrawer(graph)
        drawer.draw_graph()
