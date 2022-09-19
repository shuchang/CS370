import os
import sys
from os.path import abspath, dirname, join
sys.path.append(dirname(dirname(abspath(__file__))))
sys.path.append(join(dirname(dirname(abspath(__file__))), "framework"))

import numpy as np
import trimesh
from annoy import AnnoyIndex

from framework.viewer import Viewer
from framework.viewer.plugins import LoaderPlugin, WidgetsPlugin
from framework.viewer.widgets import MainWidget, MeshWidget, PointCloudWidget
from HW1.write_mesh import write_paraboloid_obj


def view():
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

    path = join(os.getcwd(), "HW1/mesh/paraboloid.obj")
    print("current path", path)
    viewer.load(path, False) # switch to "True" to enable "only_vertices"

    # Rendering
    viewer.launch_rendering(True)
    viewer.launch_shut()


def ann(point_p, mesh):
    f = 3 # length of item vector [x, y, z]
    n = len(mesh.vertices)
    t = AnnoyIndex(f, 'euclidean')

    for i in range(n):
        t.add_item(i, mesh.vertices[i])


    t.add_item(n, point_p)
    t.build(n_trees=20)
    t.save("test.ann")

    u = AnnoyIndex(f, 'euclidean')
    u.load("test.ann")
    result = u.get_nns_by_item(n, 2, include_distances=True)

    closest_idx = result[0][1]
    closest_distance = result[1][1]
    return closest_idx, closest_distance


def sign(point1, point2, point3):
    return ((point1[0] - point3[0])*(point2[1] - point3[1])
    - (point2[0] - point3[0])*(point1[1] - point3[1]))


def isPointInTri(point_p, point1, point2, point3):
    d1 = sign(point_p, point1, point2)
    d2 = sign(point_p, point2, point3)
    d3 = sign(point_p, point3, point1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def distance_to_faces(point_p, faces):
    planar_points = []
    for face in faces:
        point1 = face[0]
        point2 = face[1]
        point3 = face[2]
        normal = np.cross(point2-point1, point3-point1)
        c = point1[0]*normal[0] + point1[1]*normal[1] + point1[2]*normal[2]
        a = point_p[0]*normal[0] + point_p[1]*normal[1] + point_p[2]*normal[2] - c
        planar_x = point_p[0] - a*normal[0]
        planar_y = point_p[1] - a*normal[1]
        planar_z = point_p[2] - a*normal[2]
        planar_point = np.array([planar_x, planar_y, planar_z])

        # if isPointInTri(planar_point, point1, point2, point3):
        planar_points.append(planar_point)
        # else:
            # raise NotImplementedError #TODO: find projection on edge



    return np.array(planar_points)

def main():
    # file_obj = join(os.getcwd(), "HW1/mesh/paraboloid.obj")
    file_obj = join(os.getcwd(), "HW1/mesh/birdnet_quad.obj")

    if not os.path.exists(file_obj): # TODO: fix problem here
        write_paraboloid_obj(1, 1, -100, 100, -100, 100, 50, 50, file_obj)

    # mesh = om.read_trimesh(file_obj)
    mesh = trimesh.load(file_obj)
    point_p = np.array([0, 0, -1])
    print('\nfinish loading the mesh ...\n')

    closest_point_idx, distance_ann = ann(point_p, mesh)
    closest_point_xyz = mesh.vertices[closest_point_idx]
    print('closest point found by ANN: {}\n'.format(closest_point_xyz))
    print('Distance from the closest point found by ANN: {}\n'.format(distance_ann))
    print('done ANN ...\n')

    # find all faces neighboring the closest point
    neighbor_faces = np.where(mesh.faces == closest_point_idx)
    print('find {} neighboring faces:'.format(len(neighbor_faces[0])))
    neighbor_faces_idx = mesh.faces[neighbor_faces[0]]
    neighbor_faces_xyz = mesh.vertices[neighbor_faces_idx]
    project_point, project_distance = distance_to_faces(
        point_p, neighbor_faces_xyz
    )




    view() # TODO: fix view



if __name__ == "__main__":
    main()
