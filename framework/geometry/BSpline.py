import numpy
from geometry.Geometry import Geometry


class BSpline(Geometry):

    def __init__(self, control_points, knot_vector):
        super().__init__()
        # TODO: add the class data here
        self.control_points = numpy.array([])
        pass

    def point(self, t):
        # TODO
        pass

    def tangent(self, t):
        # TODO
        pass

    # noinspection PyPep8Naming
    def update_viewer_data(self, data):

        # Clear viewer
        data.clear()
        data.clear_edges()

        # Rendering settings
        curve_color = numpy.array([0.0, 0.0, 0.0, 1.0])
        polygon_color = numpy.array([0.5, 0.5, 0.5, 1.0])
        points_color = numpy.array([0.0, 0.0, 0.0, 1.0])
        num_divisions = 100
        data.point_size = 10.0
        data.line_width = 1.0
        data.show_lines = True
        data.is_visible = True

        # plot curve
        P1 = numpy.zeros((num_divisions, 3))
        P2 = numpy.zeros((num_divisions, 3))
        for i in range(num_divisions):
            t1 = float(i) / float(num_divisions)
            t2 = (float(i) + 1.0) / float(num_divisions)
            P1[i, :] = self.point(t1)
            P2[i, :] = self.point(t2)
        data.add_edges(P1, P2, numpy.array([curve_color]))

        # plot control polygon
        P1 = numpy.zeros((self.control_points.shape[0] - 1, 3))
        P2 = numpy.zeros((self.control_points.shape[0] - 1, 3))
        for i in range(self.control_points.shape[0] - 1):
            P1[i, :] = self.control_points[i, :]
            P2[i, :] = self.control_points[i, :]
        data.add_edges(P1, P2, numpy.array([polygon_color]))

        # plot control points
        data.add_points(self.control_points, numpy.array([points_color]))
