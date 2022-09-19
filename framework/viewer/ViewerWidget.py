from enum import Enum
from abc import ABCMeta, abstractmethod


class AnchorType(Enum):
    TopLeft = 0
    TopRight = 1
    BottomLeft = 2
    BottomRight = 3


class ViewerWidget(metaclass=ABCMeta):

    def __init__(self, expanded=True):
        self.viewer = None
        self.collapsed = not expanded
        self.anchor = AnchorType.TopLeft

    def init(self, viewer):
        self.viewer = viewer

    def shutdown(self):
        pass

    @abstractmethod
    def post_load(self):
        return False

    @abstractmethod
    def pre_draw(self, first):
        return False

    @abstractmethod
    def draw(self, first, scaling, x_size, y_size, x_pos, y_pos):
        return None

    @abstractmethod
    def post_draw(self, first):
        return False

    @abstractmethod
    def post_resize(self, width, height):
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
