import numpy

# Gold / Silver used in BBW / MONO / STBS / FAST
GOLD_AMBIENT = numpy.array([51.0 / 255.0, 43.0 / 255.0, 33.3 / 255.0, 1.0])
GOLD_DIFFUSE = numpy.array([255.0 / 255.0, 228.0 / 255.0, 58.0 / 255.0, 1.0])
GOLD_SPECULAR = numpy.array([255.0 / 255.0, 235.0 / 255.0, 80.0 / 255.0, 1.0])

SILVER_AMBIENT = numpy.array([0.2, 0.2, 0.2, 1.0])
SILVER_DIFFUSE = numpy.array([1.0, 1.0, 1.0, 1.0])
SILVER_SPECULAR = numpy.array([1.0, 1.0, 1.0, 1.0])

# Blue / Cyan more similar to Jovan Popovic's blue than to Mario Botsch's blue
CYAN_AMBIENT = numpy.array([59.0 / 255.0, 68.0 / 255.0, 255.0 / 255.0, 1.0])
CYAN_DIFFUSE = numpy.array([94.0 / 255.0, 185.0 / 255.0, 238.0 / 255.0, 1.0])
CYAN_SPECULAR = numpy.array([163.0 / 255.0, 221.0 / 255.0, 255.0 / 255.0, 1.0])

DENIS_PURPLE_DIFFUSE = numpy.array([80.0 / 255.0, 64.0 / 255.0, 255.0 / 255.0, 1.0])
LADISLAV_ORANGE_DIFFUSE = numpy.array([1.0, 125.0 / 255.0, 19.0 / 255.0, 0.0])

# FAST armadillos colors
FAST_GREEN_DIFFUSE = numpy.array([113.0 / 255.0, 239.0 / 255.0, 46.0 / 255.0, 1.0])
FAST_RED_DIFFUSE = numpy.array([255.0 / 255.0, 65.0 / 255.0, 46.0 / 255.0, 1.0])
FAST_BLUE_DIFFUSE = numpy.array([106.0 / 255.0, 106.0 / 255.0, 255.0 / 255.0, 1.0])
FAST_GRAY_DIFFUSE = numpy.array([150.0 / 255.0, 150.0 / 255.0, 150.0 / 255.0, 1.0])

# Basic colors
WHITE = numpy.array([255.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0, 1.0])
BLACK = numpy.array([0.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0, 1.0])
WHITE_AMBIENT = numpy.array([255.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0, 1.0])
WHITE_DIFFUSE = numpy.array([255.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0, 1.0])
WHITE_SPECULAR = numpy.array([255.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0, 1.0])
BBW_POINT_COLOR = numpy.array([239. / 255., 213. / 255., 46. / 255., 255.0 / 255.0])
BBW_LINE_COLOR = numpy.array([106. / 255., 106. / 255., 255. / 255., 255. / 255.0])
MIDNIGHT_BLUE_DIFFUSE = numpy.array([21.0 / 255.0, 27.0 / 255.0, 84.0 / 255.0, 1.0])

# Winding number colors
EASTER_RED_DIFFUSE = numpy.array([0.603922, 0.494118, 0.603922, 1.0])
WN_OPEN_BOUNDARY_COLOR = numpy.array([154. / 255., 0. / 255., 0. / 255., 1.0])
WN_NON_MANIFOLD_EDGE_COLOR = numpy.array([201. / 255., 51. / 255., 255. / 255., 1.0])

# Maya 
MAYA_GREEN = numpy.array([128. / 255., 242. / 255., 0. / 255., 1.0])
MAYA_YELLOW = numpy.array([255. / 255., 247. / 255., 50. / 255., 1.0])
MAYA_RED = numpy.array([234. / 255., 63. / 255., 52. / 255., 1.0])
MAYA_BLUE = numpy.array([0. / 255., 73. / 255., 252. / 255., 1.0])
MAYA_PURPLE = numpy.array([180. / 255., 73. / 255., 200. / 255., 1.0])
MAYA_VIOLET = numpy.array([31. / 255., 15. / 255., 66. / 255., 1.0])
MAYA_GREY = numpy.array([0.5, 0.5, 0.5, 1.0])
MAYA_CYAN = numpy.array([131. / 255., 219. / 255., 252. / 255., 1.0])
MAYA_SEA_GREEN = numpy.array([70. / 255., 252. / 255., 167. / 255., 1.0])
