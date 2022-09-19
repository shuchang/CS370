import imgui
import numpy
import quaternion
from viewer.ViewerCore import RotationType
from viewer.ViewerWidget import ViewerWidget, AnchorType


class MainWidget(ViewerWidget):
    trackball_angle = quaternion.from_rotation_matrix(numpy.identity(3))
    orthographic = True

    def __init__(self, expanded, loader):
        super().__init__(expanded)
        self.anchor = AnchorType.TopLeft
        self.loader = loader

    def post_load(self):
        return False

    def pre_draw(self, first):
        return False

    def draw(self, first, scaling, x_size, y_size, x_pos, y_pos):

        imgui.begin("Viewer", True, imgui.WINDOW_NO_SAVED_SETTINGS)

        if not first:
            imgui.set_window_position(x_pos * scaling, y_pos * scaling, imgui.ONCE)
        imgui.set_window_collapsed(self.collapsed, imgui.ONCE)

        imgui.push_item_width(imgui.get_window_width() * 0.5)

        # Viewing options
        if imgui.collapsing_header("Viewing Options", imgui.TREE_NODE_DEFAULT_OPEN):

            if imgui.button("Snap canonical view", -1, 0):
                self.viewer.snap_to_canonical_quaternion()

            # Zoom
            _, self.viewer.core().camera_zoom = imgui.drag_float(
                "Zoom", self.viewer.core().camera_zoom, 0.05, 0.1, 20.0)

            # Select rotation type
            old_type = self.viewer.core().rotation_type
            changed, new_type = imgui.combo("Camera Type", int(old_type), ["Trackball", "Two Axes", "2D Mode"])
            if changed:

                if new_type != old_type:

                    if new_type == RotationType.ROTATION_TYPE_NO_ROTATION:
                        MainWidget.trackball_angle = self.viewer.core().trackball_angle
                        MainWidget.orthographic = self.viewer.core().orthographic
                        self.viewer.core().trackball_angle = quaternion.from_rotation_matrix(numpy.identity(3))
                        self.viewer.core().orthographic = True

                    elif old_type == RotationType.ROTATION_TYPE_NO_ROTATION:
                        self.viewer.core().trackball_angle = MainWidget.trackball_angle
                        self.viewer.core().orthographic = MainWidget.orthographic

                    self.viewer.core().set_rotation_type(new_type)

            # Orthographic view
            _, self.viewer.core().orthographic = imgui.checkbox(
                "Orthographic view", self.viewer.core().orthographic)

            # Background
            changed, color = imgui.color_edit4(
                "Background",
                self.viewer.core().background_color[0],
                self.viewer.core().background_color[1],
                self.viewer.core().background_color[2],
                self.viewer.core().background_color[3],
                True)
            if changed:
                self.viewer.core().background_color = numpy.array(color)

        # Mesh
        if imgui.collapsing_header("Scene", imgui.TREE_NODE_DEFAULT_OPEN):

            if imgui.button("Load mesh##Mesh", -1, 0):
                self.viewer.open_dialog_load_mesh()

            if imgui.button("Load point cloud##Mesh", -1, 0):
                self.viewer.open_dialog_load_point_cloud()

            imgui.dummy(3.0, 0.0)
            imgui.same_line()

            if len(self.viewer.data_list) == 0:
                imgui.text("No model loaded!")
            else:
                imgui.text("Loaded models:")

            if len(self.viewer.data_list) != 0:
                for i in range(len(self.viewer.data_list)):
                    if self.loader.model(i):
                        imgui.push_id(str(i))
                        imgui.dummy(10.0, 0.0)
                        imgui.same_line()
                        active = self.viewer.selected_data_index == i
                        if active:
                            color = imgui.get_style_color_vec_4(imgui.COLOR_BUTTON_ACTIVE)
                            imgui.push_style_color(imgui.COLOR_BUTTON, color.x, color.y, color.z, color.w)
                        if imgui.button(self.loader.model(i).name, -1, 0):
                            if active:
                                self.viewer.selected_data_index = -1
                            else:
                                self.viewer.selected_data_index = i
                        if active:
                            imgui.pop_style_color()
                        imgui.pop_id()

        imgui.set_window_size(250.0 * scaling, 0.0)

        dim = imgui.Vec4(
            imgui.get_window_position().x,
            imgui.get_window_position().y,
            imgui.get_window_size().x,
            imgui.get_window_size().y)

        imgui.pop_item_width()

        imgui.end()

        return dim

    def post_draw(self, first):
        return False

    def post_resize(self, width, height):
        return False

    def mouse_down(self, button, modifier):
        return False

    def mouse_up(self, button, modifier):
        return False

    def mouse_move(self, mouse_x, mouse_y):
        return False

    def mouse_scroll(self, delta_y):
        return False

    def key_pressed(self, key, modifiers):
        return False

    def key_down(self, key, modifiers):
        return False

    def key_up(self, key, modifiers):
        return False

    def key_repeat(self, key, modifiers):
        return False
