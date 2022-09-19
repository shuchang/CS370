import os
from os.path import dirname, abspath, join
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
sys.path.append(join(dirname(dirname(abspath(__file__))), "framework"))

import numpy
from framework.viewer import Viewer
from framework.viewer.plugins import LoaderPlugin, WidgetsPlugin
from framework.viewer.widgets import MainWidget, MeshWidget, PointCloudWidget


def main():
    viewer = Viewer()

    # Change default path
    viewer.path = join(os.getcwd())

    # Attach menu plugins
    loader = LoaderPlugin()
    viewer.plugins.append(loader)
    menus = WidgetsPlugin()
    viewer.plugins.append(menus)
    menus.add(MainWidget(True, loader))
    menus.add(MeshWidget(True, loader))
    menus.add(PointCloudWidget(True, loader))

    # General drawing settings
    viewer.core().is_animating = False
    viewer.core().animation_max_fps = 30.0
    viewer.core().background_color = numpy.array([0.6, 0.6, 0.6, 1.0])

    # Initialize viewer
    if not viewer.launch_init(True, False, True, "viewer", 0, 0):
        viewer.launch_shut()
        return

    # Example of loading a model programmatically
    # path = os.path.join(os.getcwd(), '..', 'models', 'bunny10k.obj')
    path = os.path.join(os.getcwd(), "hw1/paraboloid.obj")
    # print("current path", path)
    # switch to "True" to enable "only_vertices"
    viewer.load(path, False)

    # Rendering
    viewer.launch_rendering(True)
    viewer.launch_shut()







if __name__ == "__main__":
    main()
