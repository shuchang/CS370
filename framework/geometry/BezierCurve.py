import numpy
from geometry.Geometry import Geometry


class BezierCurve(Geometry):

    def __init__(self, points):
        super().__init__()
        self.curve_degree = points.shape[0] - 1
        self.control_points = points
        pass

    def point(self, t):
        # De Casteljau's Algorithm
        temp = numpy.array(self.control_points, copy=True)
        for k in range(1, self.curve_degree + 1):
            for i in range(self.curve_degree - k + 1):
                temp[i, :] = (1 - t) * temp[i, :] + t * temp[i + 1, :]
        return temp[0, :]

    def tangent(self, t):
        # https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-der.html
        c1 = self.control_points[1:self.curve_degree + 1, :]
        for k in range(1, self.curve_degree):
            for i in range(self.curve_degree - k):
                c1[i, :] = (1 - t) * c1[i + 0, :] + t * c1[i + 1, :]

        c2 = self.control_points[0:self.curve_degree, :]
        for k in range(1, self.curve_degree):
            for i in range(self.curve_degree - k):
                c2[i, :] = (1 - t) * c2[i + 0, :] + t * c2[i + 1, :]

        return self.curve_degree * (c1[0, :] - c2[0, :])

    def length(self):
        # http://steve.hollasch.net/cgindex/curves/cbezarclen.html
        def add_if_close(points, error):
            n = points.shape[0] - 1
            polygon = 0.0
            for i in range(n):
                polygon = polygon + numpy.linalg.norm(points[i, :] - points[i + 1, :])
            cord = numpy.linalg.norm(points[0, :] - points[n, :])
            if (polygon - cord) > error:
                left, right = BezierCurve.split(points, 0.5)
                length_left = add_if_close(left, error)
                length_right = add_if_close(right, error)
                return length_left + length_right
            length = (2.0 * cord + (n - 1) * polygon) / (n + 1)
            return length

        return add_if_close(self.control_points, 0.1)

    @staticmethod
    def split(points, t):
        # https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-sub.html
        degree = points.shape[0] - 1
        left = numpy.array(points, copy=True)
        right = numpy.array(points, copy=True)
        for k in range(1, degree + 1):
            for i in range(degree - k + 1):
                right[i, :] = (1 - t) * right[i, :] + t * right[i + 1, :]
            left[k, :] = right[0, :]
        return left, right

    def split(self, t):
        return BezierCurve.split(self.control_points, t)

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
        P1 = numpy.zeros((self.curve_degree, 3))
        P2 = numpy.zeros((self.curve_degree, 3))
        for i in range(self.curve_degree):
            P1[i, :] = self.control_points[i, :]
            P2[i, :] = self.control_points[i+1, :]
        data.add_edges(P1, P2, numpy.array([polygon_color]))

        # plot control points
        data.add_points(self.control_points, numpy.array([points_color]))
