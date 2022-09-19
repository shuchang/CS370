import imgui
import numpy
from viewer.ViewerWidget import ViewerWidget, AnchorType
from geometry import PointCloud, BezierCurve, BSpline


class PointCloudWidget(ViewerWidget):

    class BreakoutException(Exception):
        pass

    def __init__(self, expanded, loader):
        super().__init__(expanded)
        self.anchor = AnchorType.TopRight
        self.loader = loader

    def post_load(self):
        return False

    def pre_draw(self, first):
        return False

    def draw(self, first, scaling, x_size, y_size, x_pos, y_pos):

        imgui.begin("PointCloud", True, imgui.WINDOW_NO_SAVED_SETTINGS)

        if not first:
            imgui.set_window_position((x_size - x_pos) * scaling, y_pos * scaling, imgui.ONCE)
        imgui.set_window_collapsed(self.collapsed, imgui.ONCE)

        imgui.push_item_width(imgui.get_window_width() * 0.5)

        if len(self.viewer.data_list) == 0:
            imgui.text("No point cloud loaded!")
        elif self.viewer.selected_data_index < 0:
            imgui.text("No point cloud selected!")
        else:

            try:
                index = self.viewer.selected_data_index
                cloud = self.loader.model(index)
                if not isinstance(cloud, PointCloud):
                    imgui.text("Object selected is not a point cloud!")
                else:

                    # Remove
                    if imgui.button("Remove##PointCloud", -1, 0):
                        self.viewer.unload()
                        raise PointCloudWidget.BreakoutException

                    imgui.spacing()

                    # Draw options
                    if imgui.collapsing_header("Draw Options", imgui.TREE_NODE_DEFAULT_OPEN):

                        if imgui.button("Center view to mesh", -1, 0):
                            self.viewer.core().align_camera_center(cloud.points, numpy.array([]))

                        changed, value = imgui.checkbox(
                            "Visible", self.viewer.core().is_set(self.viewer.data().is_visible))
                        if changed:
                            self.viewer.data().is_visible = \
                                self.viewer.core().set(self.viewer.data().is_visible, value)

                        changed, color = imgui.color_edit4(
                            "Point color",
                            cloud.point_color[0],
                            cloud.point_color[1],
                            cloud.point_color[2],
                            cloud.point_color[3],
                            True)
                        if changed:
                            cloud.point_color = numpy.array(color)
                            cloud.update_viewer_data(self.viewer.data(index))

                    imgui.spacing()

                    # BSplines
                    if imgui.collapsing_header("Project 2", imgui.TREE_NODE_DEFAULT_OPEN):

                        if imgui.button("Interpolate BSpline##PointCloud", -1, 0):
                            points = cloud.compute_bspline_interpolation()
                            # TODO: construct a BSpline instead of a Bezier curve
                            curve = BezierCurve(points)
                            curve.name = "interpolation curve"
                            self.viewer.append_data(True)
                            self.loader.model_list.append(curve)
                            curve.update_viewer_data(self.viewer.data_list[-1])

                        if imgui.button("Approximate BSpline##PointCloud", -1, 0):
                            points = cloud.compute_bspline_approximation()
                            # TODO: construct a BSpline instead of a Bezier curve
                            curve = BezierCurve(points)
                            curve.name = "approximation curve"
                            self.viewer.append_data(True)
                            self.loader.model_list.append(curve)
                            curve.update_viewer_data(self.viewer.data_list[-1])

            except PointCloudWidget.BreakoutException:
                pass

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
