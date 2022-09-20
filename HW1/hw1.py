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


def ann(point_p, mesh):
    """
    find the nearest point of point p on a mesh with ANN
    params:
        point_p: target point p
        mesh: a triangle mesh
    returns:
        nearest_idx: the idx of the nearest point
        nearest_dist: the distnace to the nearest point
    """
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

    nearest_idx = result[0][1]
    nearest_dist = result[1][1]
    return nearest_idx, nearest_dist


def sign(point1, point2, point3):
    return ((point1[0] - point3[0])*(point2[1] - point3[1])
    - (point2[0] - point3[0])*(point1[1] - point3[1]))


def isPointInTri(point_p, point1, point2, point3):
    """
    check if p is in the triangle composed of point 1, point 2 and point 3
    """
    d1 = sign(point_p, point1, point2)
    d2 = sign(point_p, point2, point3)
    d3 = sign(point_p, point3, point1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def pointToEdges(point_p, edges):
    shortest_dist = 1000

    for edge in edges:
        (point1, point2) = edge[0], edge[1]
        cos_p12 = (
            np.dot(point2 - point1, point_p - point1)/
            (np.linalg.norm(point2 - point1)*np.linalg.norm(point_p - point1))
        )
        t = np.linalg.norm(point_p - point1)*cos_p12/np.linalg.norm(point2 - point1)
        edge_point = point1 + t*(point2 - point1)
        dist = np.linalg.norm(point_p - point1)*np.sqrt(1-cos_p12**2)

        # find among three edges the one having shorest distance to p
        if dist < shortest_dist:
            shortest_dist = dist
            project_point = edge_point

    return project_point, shortest_dist


def pointToTriangles(point_p, faces):
    """
    find the projection point on faces that has the shortest distance to p
    if the projection point is not in the triangle, the closest point is on edge
    params:
        point_p: target point p
        faces: neighboring triangles of the nearest point to p on the mesh
    """
    shortest_dist = 1000

    for face in faces:
        print("\ncheck new face for the closest point ...")
        (point1, point2, point3) = face[0], face[1], face[2]
        normal = np.cross(point2 - point1, point3 - point1)
        (a, b, c) = normal[0], normal[1], normal[2]
        t = (
            (a*point1[0] + b*point1[1] + c*point1[2]) -
            (a*point_p[0] + b*point_p[1] + c*point_p[2])
        )/(a**2 + b**2 + c**2)
        planar_x = point_p[0] + a*t
        planar_y = point_p[1] + b*t
        planar_z = point_p[2] + c*t
        planar_point = np.array([planar_x, planar_y, planar_z])

        if isPointInTri(planar_point, point1, point2, point3):
            print('find projection point in the triangle')
            project_point = planar_point
            dist = np.sqrt(
                (point_p[0] - planar_x)**2 +
                (point_p[1] - planar_y)**2 +
                (point_p[2] - planar_z)**2
            )
        else:
            print('find projection point on the edge')
            edges = np.stack([
                np.stack([point1, point2], axis=0),
                np.stack([point2, point3], axis=0),
                np.stack([point3, point1], axis=0),
            ], axis=0)
            project_point, dist = pointToEdges(point_p, edges)

        # find among all faces the one having shortest distance to p
        if dist < shortest_dist:
            shortest_dist = dist
            project_point = project_point

    return project_point, shortest_dist


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
    viewer.load(path, False)
    viewer.load(join(os.getcwd(), "HW1/mesh/generated_points.obj"), True)


    # Rendering
    viewer.launch_rendering(True)
    viewer.launch_shut()


def main():
    file_obj = join(os.getcwd(), "HW1/paraboloid.obj")
    # file_obj = join(os.getcwd(), "HW1/mesh/birdnet_quad.obj")

    if not os.path.exists(file_obj): # TODO: fix problem here
        write_paraboloid_obj(1, 1, -100, 100, -100, 100, 50, 50, file_obj)

    # mesh = om.read_trimesh(file_obj)
    mesh = trimesh.load(file_obj)
    point_p = np.array([2, -1, 1])
    print('\n\nfinish loading the mesh ...\n\n')

    nearest_idx, nearest_dist = ann(point_p, mesh)
    nearest_xyz = mesh.vertices[nearest_idx]
    print('done ANN ...')
    print('nearest point of point p found by ANN: {}'.format(nearest_xyz))
    print('distance to the nearest point found by ANN: {}\n\n'.format(nearest_dist))

    # find all faces neighboring the closest point
    neighbor_faces = np.where(mesh.faces == nearest_idx)
    print('find {} neighboring faces'.format(len(neighbor_faces[0])))
    neighbor_faces_idx = mesh.faces[neighbor_faces[0]]
    neighbor_faces_xyz = mesh.vertices[neighbor_faces_idx]
    project_point, shortest_dist = pointToTriangles(
        point_p, neighbor_faces_xyz)
    print('\n\ndone point face projection ...')
    print('projection point of point p on the mesh: {}'.format(project_point))
    print('distance to the projection point: {}'.format(shortest_dist))

    # create a PointCLoud object
    projection_cloud = trimesh.points.PointCloud(
        np.vstack([project_point, point_p])
    )
    cloud_obj = join(os.getcwd(), "HW1/mesh/generated_points.obj")
    projection_cloud.export(cloud_obj)

    view()


if __name__ == "__main__":
    main()