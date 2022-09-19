import os
import numpy
import openmesh
from geometry.Geometry import Geometry


class PointCloud(Geometry):

    def __init__(self):
        super().__init__()
        self.point_color = numpy.array([0.0, 0.0, 0.0, 1.0])
        self.points = numpy.empty((0, 3))

    def load(self, filename):
        # Load as mesh
        try:
            mesh = openmesh.read_polymesh(
                filename, binary=False, msb=False, lsb=False, swap=False,
                vertex_normal=False, vertex_color=False, vertex_tex_coord=False, halfedge_tex_coord=False,
                edge_color=False, face_normal=False, face_color=False, face_texture_index=False,
                color_alpha=False, color_float=False)
        except RuntimeError as error:
            print("Error:", error)
            return False
        if mesh is None:
            print("Error: Error loading point cloud from file ", filename)
            return False

        # Point cloud name
        self.name = os.path.splitext(os.path.basename(filename))[0]

        # Extract vertices
        self.points = mesh.points()

        # Success!
        return True

    def save(self, filename):
        # Not implemented
        return False

    def update_viewer_data(self, data):

        # Clear viewer
        data.clear()
        data.clear_edges()
        data.set_mesh(self.points, numpy.empty((0, 3)))

        # Paint points
        data.add_points(self.points, numpy.array([self.point_color]))
        data.point_size = 5.0

    def point_color(self):
        return self.point_color

    def compute_bspline_interpolation(self):
        # TODO
        points = numpy.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.0],
            [0.2, 0.1, 0.0],
            [0.3, 0.0, 0.0]
        ])
        return points

    def compute_bspline_approximation(self):
        # TODO
        points = numpy.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.0],
            [0.2, 0.1, 0.0],
            [0.3, 0.0, 0.0]
        ])
        return points
