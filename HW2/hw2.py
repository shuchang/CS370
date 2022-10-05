import os
import sys
from os.path import abspath, dirname, join
sys.path.append(dirname(dirname(abspath(__file__))))
sys.path.append(join(dirname(dirname(abspath(__file__))), "framework"))

import numpy as np

from framework.viewer import Viewer
from framework.viewer.plugins import LoaderPlugin, WidgetsPlugin
from framework.viewer.widgets import MainWidget, MeshWidget, PointCloudWidget


def view(file_obj):
    """Load the mesh, point_p and projection point to viewer"""
    viewer = Viewer()

    # Change default path
    viewer.path = os.getcwd()

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
    viewer.core().background_color = np.array([0.6, 0.6, 0.6, 1.0])

    # Initialize viewer
    if not viewer.launch_init(True, False, True, "viewer", 0, 0):
        viewer.launch_shut()
        return

    viewer.load(file_obj, False)

    # Rendering
    viewer.launch_rendering(True)
    viewer.launch_shut()


def main():
    file_obj = join(os.getcwd(), "HW2/ExerciseData/model_1.obj")
    view(file_obj)


if __name__ == "__main__":
    main()