import imgui
import numpy
from viewer.ViewerWidget import ViewerWidget, AnchorType
from viewer.opengl.MeshGL import DirtyFlags
from geometry.Mesh import Mesh, NoiseDirection


class MeshWidget(ViewerWidget):

    class BreakoutException(Exception):
        pass

    noiseStandardDeviation = 0.1
    noiseDirection = NoiseDirection.NORMAL
    curvatureNeighbours = 6

    def __init__(self, expanded, loader):
        super().__init__(expanded)
        self.anchor = AnchorType.TopRight
        self.loader = loader

    def post_load(self):
        return False

    def pre_draw(self, first):
        return False

    def draw(self, first, scaling, x_size, y_size, x_pos, y_pos):

        imgui.begin("Mesh", True, imgui.WINDOW_NO_SAVED_SETTINGS)

        if not first:
            imgui.set_window_position((x_size - x_pos) * scaling, y_pos * scaling, imgui.ONCE)
        imgui.set_window_collapsed(self.collapsed, imgui.ONCE)

        imgui.push_item_width(imgui.get_window_width() * 0.5)

        if len(self.viewer.data_list) == 0:
            imgui.text("No mesh loaded!")
        elif self.viewer.selected_data_index < 0:
            imgui.text("No mesh selected!")
        else:

            try:
                index = self.viewer.selected_data_index
                color = numpy.array([1, 0, 0]).reshape(1, 3)
                self.viewer.data(index).clear_edges()
                self.viewer.data(index).clear_points()
                self.viewer.data(index).add_points(self.viewer.point_p.reshape((1,3)), color)
                self.viewer.data(index).add_edges(
                    self.viewer.point_p.reshape((1,3)),
                    self.viewer.project_point.reshape((1,3)), color
                )
                mesh = self.loader.model(index)
                if not isinstance(mesh, Mesh):
                    imgui.text("Object selected is not a mesh!")
                else:
                    # Reset
                    if imgui.button("Reset##Mesh", -1, 0):
                        mesh.reset_mesh()
                        mesh.update_viewer_data(self.viewer.data(index))

                    # Save
                    if imgui.button("Save##Mesh", -1, 0):
                        self.viewer.open_dialog_save_mesh()

                    # Remove
                    if imgui.button("Remove##Mesh", -1, 0):
                        self.viewer.unload()
                        raise MeshWidget.BreakoutException

                    imgui.spacing()

                    # Draw options
                    if imgui.collapsing_header("Draw Options", imgui.TREE_NODE_DEFAULT_OPEN):

                        if imgui.button("Center view to mesh", -1, 0):
                            self.viewer.core().align_camera_center(self.viewer.data().V, self.viewer.data().F)

                        if imgui.button("Normalize mesh", -1, 0):
                            mesh.normalize()
                            mesh.update_viewer_data(self.viewer.data(index))

                        changed, value = imgui.checkbox(
                            "Visible", self.viewer.core().is_set(self.viewer.data().is_visible))
                        if changed:
                            self.viewer.data().is_visible = \
                                self.viewer.core().set(self.viewer.data().is_visible, value)

                        changed, self.viewer.data().face_based = imgui.checkbox(
                            "Face-based", self.viewer.data().face_based)
                        if changed:
                            self.viewer.data().dirty = DirtyFlags.DIRTY_ALL

                        changed, mesh.render_flat_faces = imgui.checkbox(
                            "Render as flat triangles", mesh.render_flat_faces)
                        if changed:
                            mesh.update_viewer_data(self.viewer.data(index))

                        changed, value = imgui.checkbox(
                            "Show texture", self.viewer.core().is_set(self.viewer.data().show_texture))
                        if changed:
                            self.viewer.data().show_texture = \
                                self.viewer.core().set(self.viewer.data().show_texture, value)

                        changed, self.viewer.data().invert_normals = imgui.checkbox(
                            "Invert normals", self.viewer.data().invert_normals)
                        if changed:
                            self.viewer.data().dirty |= DirtyFlags.DIRTY_NORMAL

                        # changed, value = imgui.checkbox(
                        #     "Show overlay", self.viewer.core().is_set(self.viewer.data().show_overlay))
                        # if changed:
                        #     self.viewer.data().show_overlay = \
                        #         self.viewer.core().set(self.viewer.data().show_overlay, value)

                        # changed, value = imgui.checkbox(
                        #     "Show overlay depth", self.viewer.core().is_set(self.viewer.data().show_overlay_depth))
                        # if changed:
                        #     self.viewer.data().show_overlay_depth = \
                        #         self.viewer.core().set(self.viewer.data().show_overlay_depth, value)

                        changed, color = imgui.color_edit4(
                            "Line color",
                            mesh.edge_color[0],
                            mesh.edge_color[1],
                            mesh.edge_color[2],
                            mesh.edge_color[3],
                            True)
                        if changed:
                            mesh.edge_color = numpy.array(color)
                            mesh.update_viewer_data(self.viewer.data(index))

                        imgui.push_item_width(imgui.get_window_width() * 0.3)
                        _, self.viewer.data().shininess = imgui.drag_float(
                            "Shininess", self.viewer.data().shininess, 0.05, 0.0, 100.0)
                        imgui.pop_item_width()

                    imgui.spacing()

                    # Noise
                    if imgui.collapsing_header("Noise", imgui.TREE_NODE_DEFAULT_OPEN):

                        _, MeshWidget.noiseStandardDeviation = imgui.drag_float(
                            "sigma", MeshWidget.noiseStandardDeviation, 0.01, 0.0, 5.0, "%.2f")

                        _, MeshWidget.noiseDirection = imgui.combo(
                            "Direction", int(MeshWidget.noiseDirection), ["Normals", "Random"])

                        if imgui.button("Add Gaussian noise", -1, 0):
                            mesh.noise(MeshWidget.noiseStandardDeviation, MeshWidget.noiseDirection)
                            mesh.update_viewer_data(self.viewer.data(index))

                    imgui.spacing()

                    # Overlays
                    if imgui.collapsing_header("Overlays", imgui.TREE_NODE_DEFAULT_OPEN):

                        changed, mesh.render_edges = imgui.checkbox(
                            "Wireframe", mesh.render_edges)
                        if changed:
                            mesh.update_viewer_data(self.viewer.data(index))

                        changed, value = imgui.checkbox(
                            "Fill", self.viewer.core().is_set(self.viewer.data().show_faces))
                        if changed:
                            self.viewer.data().show_faces = \
                                self.viewer.core().set(self.viewer.data().show_faces, value)

                        changed, value = imgui.checkbox(
                            "Show vertex labels", self.viewer.core().is_set(self.viewer.data().show_vertex_labels))
                        if changed:
                            self.viewer.data().show_vertex_labels = \
                                self.viewer.core().set(self.viewer.data().show_vertex_labels, value)

                        changed, value = imgui.checkbox(
                            "Show faces labels", self.viewer.core().is_set(self.viewer.data().show_face_labels))
                        if changed:
                            self.viewer.data().show_face_labels = \
                                self.viewer.core().set(self.viewer.data().show_face_labels, value)

                        changed, value = imgui.checkbox(
                            "Show extra labels", self.viewer.core().is_set(self.viewer.data().show_custom_labels))
                        if changed:
                            self.viewer.data().show_custom_labels = \
                                self.viewer.core().set(self.viewer.data().show_custom_labels, value)

                    imgui.spacing()

                    # Subdivision
                    if imgui.collapsing_header("Subdivision", imgui.TREE_NODE_DEFAULT_OPEN):

                        if imgui.button("Catmull Clark##Mesh", -1, 0):
                            # Initialize subdivider
                            # OpenMesh::Subdivider::Uniform::CatmullClarkT<OpenMesh::Mesh> catmull;
                            # Execute 3 subdivision steps
                            # catmull.attach(mLoader->mesh(index)->mesh())
                            # catmull(3)
                            # catmull.detach()
                            mesh.update_viewer_data(self.viewer.data(index))

                    # Curvatures
                    if imgui.collapsing_header("Project 1", imgui.TREE_NODE_DEFAULT_OPEN):

                        _, MeshWidget.curvatureNeighbours = imgui.drag_int(
                            "Num. neighbours", MeshWidget.curvatureNeighbours, 1.0, 6, 25, "%d")

                        if imgui.button("Paint Gaussian Curvatures##Mesh", -1, 0):
                            curvatures, _ = mesh.curvatures(k=MeshWidget.curvatureNeighbours)
                            self.viewer.data(index).set_data(curvatures)

                        if imgui.button("Paint Mean Curvatures##Mesh", -1, 0):
                            _, curvatures = mesh.curvatures(k=MeshWidget.curvatureNeighbours)
                            self.viewer.data(index).set_data(curvatures)

                        if imgui.button("Paint Asymptotic Directions##Mesh", -1, 0):
                            indices, directions = mesh.asymptotic_directions()
                            num_directions = len(indices)
                            src = self.viewer.data(index).V[indices, :] - 0.5 * directions
                            dst = self.viewer.data(index).V[indices, :] + 0.5 * directions
                            color = numpy.array([1, 0, 0])
                            colors = numpy.repeat([color], num_directions, axis=0)
                            self.viewer.data(index).add_edges(src, dst, colors)

            except MeshWidget.BreakoutException:
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
