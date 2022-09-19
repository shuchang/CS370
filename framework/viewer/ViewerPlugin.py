from abc import ABCMeta, abstractmethod


class ViewerPlugin(metaclass=ABCMeta):

    def __init__(self):
        self.viewer = None
        self.name = "dummy"

    def init(self, viewer):
        self.viewer = viewer

    def shutdown(self):
        pass

    @abstractmethod
    def load(self, filename, only_vertices):
        return False

    @abstractmethod
    def unload(self):
        return False

    @abstractmethod
    def save(self, filename, only_vertices):
        return False

    @abstractmethod
    def serialize(self, buffer):
        return False

    @abstractmethod
    def deserialize(self, buffer):
        return False

    @abstractmethod
    def post_load(self):
        return False

    @abstractmethod
    def pre_draw(self, first):
        return False

    @abstractmethod
    def post_draw(self, first):
        return False

    @abstractmethod
    def post_resize(self, w, h):
        return False

    @abstractmethod
    def mouse_down(self, button, modifier):
        return False

    @abstractmethod
    def mouse_up(self, button, modifier):
        return False

    @abstractmethod
    def mouse_move(self, mouse_x, mouse_y):
        return False

    @abstractmethod
    def mouse_scroll(self, delta_y):
        return False

    @abstractmethod
    def key_pressed(self, key, modifiers):
        return False

    @abstractmethod
    def key_down(self, key, modifiers):
        return False

    @abstractmethod
    def key_up(self, key, modifiers):
        return False

    @abstractmethod
    def key_repeat(self, key, modifiers):
        return False
