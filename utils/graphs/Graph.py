from dataclasses import dataclass


@dataclass
class Graph:
    def __init__(self, size, title, x_label, y_label, show=True):
        self.size = size
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.show = show
