from abc import ABCMeta, abstractmethod


class Geometry(metaclass=ABCMeta):
    counter = 0

    def __init__(self):
        self.id = Geometry.counter
        Geometry.counter += 1
        self.name = ""
        pass

    def name(self):
        return self.name

    def id(self):
        return self.id

    @abstractmethod
    def update_viewer_data(self, data):
        pass
