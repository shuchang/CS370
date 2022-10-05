import copy
import os
import math
import numpy
import openmesh
from enum import IntEnum
from utils.colormap import ColorMapType
from utils.maths import gaussian, random_vector
from geometry.Geometry import Geometry
import hnswlib
import pdb
import numpy as np
import scipy.optimize
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class NoiseDirection(IntEnum):
    NORMAL = 0
    RANDOM = 1


class Mesh(Geometry):

    def __init__(self):
        super().__init__()
        self.render_edges = True
        self.render_flat_faces = False
        self.edge_color = numpy.array([0.0, 0.0, 0.0, 1.0])
        self.colormap = ColorMapType.NUM_COLOR_MAP_TYPES
        self.mesh = None
        self.mesh_original = None

    def load(self, filename):
        # Load a mesh
        try:
            self.mesh = openmesh.read_polymesh(
                filename, binary=False, msb=False, lsb=False, swap=False,
                vertex_normal=False, vertex_color=False, vertex_tex_coord=False, halfedge_tex_coord=False,
                edge_color=False, face_normal=False, face_color=False, face_texture_index=False,
                color_alpha=False, color_float=False)
        except RuntimeError as error:
            print("Error:", error)
            return False
        if self.mesh is None:
            print("Error: Error loading mesh from file ", filename)
            return False
        self.mesh.request_vertex_texcoords2D()

        # Mesh name
        self.name = os.path.splitext(os.path.basename(filename))[0]

        # We need normals
        self.mesh.request_face_normals()
        self.mesh.request_vertex_normals()
        self.mesh.update_normals()

        # Save original mesh
        self.mesh_original = copy.deepcopy(self.mesh)

        # Success!
        return True

    def save(self, filename):
        # Save a mesh
        openmesh.write_mesh(self.mesh, filename)
        # if False:
        #     print("Error: Error saving mesh to file ", filename)
        #     return False
        # Success!
        return True

    def update_viewer_data(self, data):

        # Clear viewer
        data.clear()
        data.clear_edges()

        # Convert mesh to viewer format
        tmp_v, tmp_f, tmp_f_to_f, tmp_n, tmp_uv, p1, p2 = Mesh.to_render_data(self.mesh)

        # Plot the mesh
        data.set_mesh(tmp_v, tmp_f)
        data.FtoF = tmp_f_to_f
        data.set_normals(tmp_n)
        data.set_uv(tmp_uv)
        if self.render_flat_faces:
            data.compute_normals()
        else:
            data.face_based = False
        if self.render_edges:
            data.add_edges(p1, p2, numpy.array([self.edge_color]))
        data.line_width = 1.0
        data.show_lines = False
        # show_texture = True
        data.show_texture = False

        # Colors
        data.uniform_colors([51.0 / 255.0, 43.0 / 255.0, 33.3 / 255.0, 255.0],
                            [255.0 / 255.0, 228.0 / 255.0, 58.0 / 255.0, 255.0],
                            [255.0 / 255.0, 235.0 / 255.0, 80.0 / 255.0, 255.0])

    @staticmethod
    def to_render_data(mesh: openmesh.PolyMesh):
        # Triangulate
        face_map = dict()
        normals_orig = dict()
        tri_mesh = copy.deepcopy(mesh)
        for fh in tri_mesh.faces():
            try:
                face_map[fh.idx()]
            except KeyError:
                face_map[fh.idx()] = fh.idx()

            n = tri_mesh.normal(fh)
            try:
                normals_orig[fh.idx()]
            except KeyError:
                normals_orig[fh.idx()] = n

            base_heh = tri_mesh.halfedge_handle(fh)
            start_vh = tri_mesh.from_vertex_handle(base_heh)
            prev_heh = tri_mesh.prev_halfedge_handle(base_heh)
            next_heh = tri_mesh.next_halfedge_handle(base_heh)

            while tri_mesh.to_vertex_handle(tri_mesh.next_halfedge_handle(next_heh)) != start_vh:

                next_next_heh = tri_mesh.next_halfedge_handle(next_heh)

                new_fh = tri_mesh.new_face()
                tri_mesh.set_halfedge_handle(new_fh, base_heh)

                face_map[new_fh.idx()] = fh.idx()

                normals_orig[new_fh.idx()] = n

                new_heh = tri_mesh.new_edge(tri_mesh.to_vertex_handle(next_heh), start_vh)

                tri_mesh.set_next_halfedge_handle(base_heh, next_heh)
                tri_mesh.set_next_halfedge_handle(next_heh, new_heh)
                tri_mesh.set_next_halfedge_handle(new_heh, base_heh)

                tri_mesh.set_face_handle(base_heh, new_fh)
                tri_mesh.set_face_handle(next_heh, new_fh)
                tri_mesh.set_face_handle(new_heh, new_fh)

                tri_mesh.copy_all_properties(prev_heh, new_heh, True)
                tri_mesh.copy_all_properties(prev_heh, tri_mesh.opposite_halfedge_handle(new_heh), True)
                tri_mesh.copy_all_properties(fh, new_fh, True)

                base_heh = tri_mesh.opposite_halfedge_handle(new_heh)
                next_heh = next_next_heh

            tri_mesh.set_halfedge_handle(fh, base_heh)  # the last face takes the handle _fh
            tri_mesh.set_next_halfedge_handle(base_heh, next_heh)
            tri_mesh.set_next_halfedge_handle(tri_mesh.next_halfedge_handle(next_heh), base_heh)
            tri_mesh.set_face_handle(base_heh, fh)

        # Resize arrays
        verts = numpy.empty((tri_mesh.n_vertices(), 3))
        faces = numpy.empty((tri_mesh.n_faces(), 3), dtype=numpy.uint32)
        f_to_f = numpy.empty((tri_mesh.n_faces(), 1), dtype=numpy.uint32)
        norms = numpy.empty((tri_mesh.n_faces(), 3))
        if mesh.has_vertex_texcoords2D():
            texs = numpy.empty((tri_mesh.n_vertices(), 2))
        else:
            texs = None

        # Vertices
        for vh in tri_mesh.vertices():
            p = tri_mesh.point(vh)
            verts[vh.idx(), 0] = p[0]
            verts[vh.idx(), 1] = p[1]
            verts[vh.idx(), 2] = p[2]

        # Faces
        for fh in tri_mesh.faces():
            vi = 0
            for fvi in tri_mesh.fv(fh):
                faces[fh.idx(), vi] = fvi.idx()
                vi += 1

        # Face map
        for key, value in face_map.items():
            f_to_f[key, 0] = value

        # Normals
        for key, value in normals_orig.items():
            n = value
            norms[key, 0] = n[0]
            norms[key, 1] = n[1]
            norms[key, 2] = n[2]

        # TexCoords
        if mesh.has_vertex_texcoords2D():
            for vh in tri_mesh.vertices():
                tex = tri_mesh.texcoord2D(vh)
                texs[vh.idx(), 0] = tex[0]
                texs[vh.idx(), 1] = tex[1]

        # Edges
        edges1 = numpy.empty((mesh.n_edges(), 3))
        edges2 = numpy.empty((mesh.n_edges(), 3))
        for eh in mesh.edges():
            vh1 = mesh.to_vertex_handle(mesh.halfedge_handle(eh, 0))
            vh2 = mesh.from_vertex_handle(mesh.halfedge_handle(eh, 0))
            v1 = mesh.point(mesh.vertex_handle(vh1.idx()))
            v2 = mesh.point(mesh.vertex_handle(vh2.idx()))

            edges1[eh.idx(), 0] = v1[0]
            edges1[eh.idx(), 1] = v1[1]
            edges1[eh.idx(), 2] = v1[2]
            edges2[eh.idx(), 0] = v2[0]
            edges2[eh.idx(), 1] = v2[1]
            edges2[eh.idx(), 2] = v2[2]

        return verts, faces, f_to_f, norms, texs, edges1, edges2

    def mesh(self):
        return self.mesh

    def clean(self):
        pass

    def reset_mesh(self):
        self.mesh = copy.deepcopy(self.mesh_original)
        self.clean()

    def set_mesh(self, mesh):
        self.mesh = copy.deepcopy(mesh)
        self.clean()

    def render_flat_faces(self):
        return self.render_flat_faces

    def render_edges(self):
        return self.render_edges

    def edge_color(self):
        return self.edge_color

    def colormap(self):
        return self.colormap

    def num_vertices(self):
        return self.mesh.n_vertices()

    def vertex(self, index):
        p = self.mesh.point(self.mesh.vertex_handle(index))
        return p

    def set_vertex(self, index, p):
        self.mesh.set_point(self.mesh.vertex_handle(index), p)

    def face_center(self, index):
        fh = self.mesh.face_handle(index)
        p = numpy.array([0.0, 0.0, 0.0])
        for vh in self.mesh.fv(fh):
            p += self.mesh.point(vh)
        p /= self.mesh.valence(fh)
        return p

    def set_face_center(self, index, v):
        t = v - self.face_center(index)
        fh = self.mesh.face_handle(index)
        for vh in self.mesh.fv(fh):
            self.mesh.set_point(vh, self.mesh.point(vh) + t)

    def normalize(self):
        total_area = 0.0
        barycenter = [numpy.array([0.0, 0.0, 0.0])] * self.mesh.n_faces()
        area = [0.0] * self.mesh.n_faces()

        # loop over faces
        for fh in self.mesh.faces():

            # compute barycenter of face
            center = numpy.array([0.0, 0.0, 0.0])
            valence = 0
            vertices = []
            for vh in self.mesh.fv(fh):
                center += self.mesh.point(vh)
                valence += 1
                vertices.append(vh)
            barycenter[fh.idx()] = center / valence

            # compute area of face
            if valence == 3:
                v0 = self.mesh.point(vertices[0])
                v1 = self.mesh.point(vertices[1])
                v2 = self.mesh.point(vertices[2])

                # A = 0.5 * || (v0 - v1) x (v2 - v1) ||
                a = 0.5 * numpy.linalg.norm(numpy.cross((v0 - v1), (v2 - v1)))
                area[fh.idx()] = a
                total_area += area[fh.idx()]

            elif valence == 4:
                v0 = self.mesh.point(vertices[0])
                v1 = self.mesh.point(vertices[1])
                v2 = self.mesh.point(vertices[2])
                v3 = self.mesh.point(vertices[3])

                # A = 0.5 * || (v0 - v1) x (v2 - v1) ||
                a012 = numpy.linalg.norm(numpy.cross((v0 - v1), (v2 - v1)))
                a023 = numpy.linalg.norm(numpy.cross((v0 - v2), (v3 - v2)))
                a013 = numpy.linalg.norm(numpy.cross((v0 - v1), (v3 - v1)))
                a123 = numpy.linalg.norm(numpy.cross((v1 - v2), (v3 - v2)))
                area[fh.idx()] = (a012 + a023 + a013 + a123) * 0.25
                total_area += area[fh.idx()]

            else:
                print("Error: Arbitrary polygonal faces not supported")
                return

        # compute mesh centroid
        centroid = numpy.array([0.0, 0.0, 0.0])
        for i in range(self.mesh.n_faces()):
            centroid += area[i] / total_area * barycenter[i]

        # normalize mesh
        for vh in self.mesh.vertices():
            p = self.mesh.point(vh)
            p -= centroid  # subtract centroid (important for numerics)
            p /= math.sqrt(total_area)  # normalize to unit surface area (important for numerics)
            self.mesh.set_point(vh, p)

    def average_edge_length(self):
        sum_edge_length = 0.0
        for eh in self.mesh.edges():
            sum_edge_length += self.mesh.calc_edge_length(eh)
        return sum_edge_length / self.mesh.n_edges()

    def average_dihedral_angle(self):
        sum_dihedral_angle = 0.0
        for eh in self.mesh.edges():
            if self.mesh.is_boundary(eh):
                sum_dihedral_angle += self.mesh.calc_dihedral_angle(eh)
        return sum_dihedral_angle / self.mesh.n_edges()

    def noise(self, standard_deviation, noise_direction):

        average_length = self.average_edge_length()

        if noise_direction == NoiseDirection.NORMAL:

            for vh in self.mesh.vertices():
                n = self.mesh.normal(vh)
                g = gaussian(0, average_length * standard_deviation)
                p = self.mesh.point(vh) + n * g
                self.mesh.set_point(vh, p)

        elif noise_direction == NoiseDirection.RANDOM:

            for vh in self.mesh.vertices():
                n = random_vector()
                g = gaussian(0, average_length * standard_deviation)
                p = self.mesh.point(vh) + n * g
                self.mesh.set_point(vh, p)

    def curvatures(self, k=6):
        # TODO: compute curvatures
        # Return: Gauss curvatures and Mean curvatures
        # Reference: https://github.com/alecjacobson/geometry-processing-curvature
        gaussian_curvatures_list = []
        mean_curvatures_list = []

        def cal_rotation_matrix(vec1, vec2):
            # Reference https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
            """ Find the rotation matrix that aligns vec1 to vec2
            :param vec1: A 3d "source" vector
            :param vec2: A 3d "destination" vector
            :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
            """
            a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
            v = np.cross(a, b)
            c = np.dot(a, b)
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
            return rotation_matrix

        for vh in tqdm(self.mesh.vertices()):
            # get the neighbors points
            neighbors = []
            while len(neighbors) < k:
                if len(neighbors) == 0:
                    nhop_list = [vh]
                # share edge of this point
                tem = []
                for item in nhop_list:
                    for neighbor_nhop_plus1 in self.mesh.vv(item):
                        if neighbor_nhop_plus1 not in neighbors:
                            neighbors.append(neighbor_nhop_plus1)
                            tem.append(neighbor_nhop_plus1)
                nhop_list = tem
            # get the first k neighbors
            neighbors = neighbors[:k]

            # find nearest neighbors
            # distance = []
            # for vh_nei in self.mesh.vertices():
            #     distance.append((vh_nei, numpy.linalg.norm(numpy.array(self.mesh.point(vh)) - numpy.array(self.mesh.point(vh_nei)))))
            # distance = sorted(distance, key=lambda x: x[1], reverse=False)
            # neighbors = [i[0] for i in distance[1:k+1]]

            point_array = []
            for item in neighbors:
                point_array.append(self.mesh.point(item))  #x,y,z

            # calculate the normals and the rotation matrix to (0, 0, 1)
            normal_vh = self.mesh.normal(vh)
            rotation_matrix = cal_rotation_matrix(normal_vh, [0, 0, 1])
            rotated_point_array = []
            for p in point_array:
                x = p[0]
                y = p[1]
                z = p[2]
                vec = np.array([x, y, z])
                tem = numpy.dot(rotation_matrix, (vec - self.mesh.point(vh)))
                rotated_point_array.append(tem)

            # Reference:
            # https://blog.csdn.net/m0_37957160/article/details/110391924
            # B = A * coff  => coff = A-1 * B
            A = np.zeros((6, 6))
            B = np.zeros((6, 1))
            a = b = c = d = e = f = 0

            for p in rotated_point_array:
                x = p[0]
                y = p[1]
                z = p[2]

                add_matrix = [
                    [np.power(x, 4), np.power(x, 3) * y, np.power(x, 2) * np.power(y, 2), np.power(x, 3), np.power(x, 2) * y, np.power(x, 2)],
                    [0, np.power(x, 2) * np.power(y, 2), x * np.power(y, 3), np.power(x,2)*y, x*np.power(y,2), x*y],
                    [0, 0, np.power(y,4), x*np.power(y,2), np.power(y,3), np.power(y,2)],
                    [0, 0, 0, np.power(x,2), x*y, x],
                    [0, 0, 0, 0, np.power(y, 2), y],
                    [0, 0, 0, 0, 0, 1]
                ]
                for i in range(6):
                    for j in range(6):
                        if i > j:
                            add_matrix[i][j] = add_matrix[j][i]
                A += add_matrix

                add_matrix_B = [
                    np.power(x,2)*z, x*y*z, np.power(y,2)*z,
                    x*z, y*z, z
                ]
                add_matrix_B = np.array(add_matrix_B).reshape((6,1))
                B += add_matrix_B
                # calculate the coefficient by coff = A-1 * B
                # TODO pinv

            try:
                [a, b, c, d, e, f] = np.linalg.solve(A, B).flatten()
            except:
                [a, b, c, d, e, f] = np.matmul(np.linalg.pinv(A), B).flatten()

            # uv on the vertice
            target_point = self.mesh.point(vh)
            u = 2 * a * 0 + b * 0 + d
            v = 2 * c * 0 + b * 0 + e

            # After getting the coefficient, calcualte the EFGefg
            # # EFG  first fundamental forms
            E = 1 + u ** 2
            F = u * v
            G = 1 + v ** 2
            # second fundamental forms
            common = np.sqrt(u**2 + v**2 + 1)
            L = (2 * a) / common
            M = b / common
            N = (2 * c) / common

            gauss_vh = (L * N - np.power(M,2)) / (E * G - np.power(F, 2))
            mean_vh = (E * N - 2 * F * M + G * L) / (2 * E * G - 2 * np.power(F,2))

            gaussian_curvatures_list.append(gauss_vh)
            mean_curvatures_list.append(mean_vh)
            # # valid

        # gauss = numpy.random.rand(self.mesh.n_vertices(), 1)
        # mean = numpy.random.rand(self.mesh.n_vertices(), 1)
        gaussian_curvatures_list = np.array(gaussian_curvatures_list, dtype=float)
        mean_curvatures_list = np.array(mean_curvatures_list, dtype=float)
        # print(np.corrcoef(gaussian_curvatures_list, mean_curvatures_list))
        res = stats.pearsonr(gaussian_curvatures_list, mean_curvatures_list)
        print(res)
        # print(gaussian_curvatures_list[:10])
        # print(mean_curvatures_list[:10])
        plt.scatter(gaussian_curvatures_list, mean_curvatures_list)
        plt.xlabel("gaussian curvature")
        plt.ylabel("mean curvature")
        plt.show()
        sns.kdeplot(gaussian_curvatures_list)
        plt.show()
        sns.kdeplot(mean_curvatures_list)
        plt.show()
        gaussian_curvatures_list = gaussian_curvatures_list.reshape((self.mesh.n_vertices(), 1))
        mean_curvatures_list = mean_curvatures_list.reshape((self.mesh.n_vertices(), 1))
        return gaussian_curvatures_list, mean_curvatures_list

    def asymptotic_directions(self):
        # TODO: compute asymptotic directions
        indices = numpy.arange(self.mesh.n_vertices())
        normals = self.mesh.vertex_normals()
        return indices, normals

    def cal_nearest_point(self, query):
        # ANN Search
        # Mesh to ANN as tree
        points_list = []
        points_vars = []
        for vh in self.mesh.vertices():
            points_vars.append(vh)
            p = self.mesh.point(vh)
            points_list.append(p)
        points_list = np.array(points_list)
        indexs = hnswlib.Index(space = 'l2', dim = 3) # possible options are l2, cosine or ip
        indexs.init_index(max_elements=points_list.shape[0])
        indexs.add_items(points_list, np.arange(points_list.shape[0]))
        labels, distances = indexs.knn_query(query, k = 1)
        print(labels)
        print(distances)

        nearest_point_in_mesh = points_vars[labels[0][0]]
        face_array = self.mesh.face_vertex_indices()

        # Mapping to neighbors triangle project
        distance_list = []
        candidate_points = []

        for fh in self.mesh.vf(nearest_point_in_mesh):
            face_index = fh.idx()
            points_index = face_array[face_index]

            point0 = points_list[points_index[0]]   # A
            point1 = points_list[points_index[1]]   # B
            point2 = points_list[points_index[2]]   # C

            # find the normal to the triangle
            # Reference: https://math.stackexchange.com/questions/588871/minimum-distance-between-point-and-face
            n = np.cross((point1 - point0), (point2 - point0))
            normalized_n = n / np.sqrt(np.sum(n**2))

            # calculate the projected point
            t = np.dot(point0 - query, normalized_n)
            projected_point = query + t * normalized_n
            distance = np.sqrt(np.sum((query - projected_point) ** 2))

            # judge whether the projected point is in the triangle
            # using Barycentric coordinate system
            # Reference: https://www.pianshen.com/article/48521133266/
            # https://blog.csdn.net/charlee44/article/details/117629956
            # p = (1-u-v)*point0 + u*point1 + v*point2
            v0 = point2 - point0
            v1 = point1 - point0
            v2 = query - point0

            v0v0 = np.dot(v0, v0)
            v0v1 = np.dot(v0, v1)
            v0v2 = np.dot(v0, v2)
            v1v1 = np.dot(v1, v1)
            v1v2 = np.dot(v1, v2)
            D = v0v0 * v1v1 - v0v1 * v0v1

            # thus u and v is
            u = float((v1v1 * v0v2 - v0v1 * v1v2) / D)
            v = float((v0v0 * v1v2 - v0v1 * v0v2) / D)
            # if u>=0 v>=0 u+v<=1, return
            if u >= 0 and v >= 0 and u+v <= 1:
                distance_list.append(distance)
                candidate_points.append(projected_point)
            # if no, calculate
            else:
                # calculate the distance from the query point to the edge
                # distance to each edge
                dist_edge1, target_point1 = self.cal_point_to_line_distance((point2, point1), query)
                dist_edge2, target_point2 = self.cal_point_to_line_distance((point2, point0), query)
                dist_edge3, target_point3 = self.cal_point_to_line_distance((point1, point0), query)
                # distance to each point
                dist_point1, target_point4 = np.linalg.norm(point0 - query), point0
                dist_point2, target_point5 = np.linalg.norm(point1 - query), point1
                dist_point3, target_point6 = np.linalg.norm(point2 - query), point2
                # select min
                tem_dist_list = [dist_edge1, dist_edge2, dist_edge3, dist_point1, dist_point2, dist_point3]
                tem_candidate_points = [target_point1, target_point2, target_point3, target_point4,
                                     target_point5, target_point6]
                min_dist = min(tem_dist_list)
                min_index = tem_dist_list.index(min_dist)
                target_point = tem_candidate_points[min_index]

                # assign for this face
                distance_list.append(min_dist)
                candidate_points.append(target_point)

        print(distance_list)
        print(candidate_points)
        min_distance = min(distance_list)
        min_distance_idx = distance_list.index(min_distance)
        nearest_point = candidate_points[min_distance_idx]
        return nearest_point, min_distance

    def cal_point_to_line_distance(self, line, query):
        begin = line[0]
        end = line[1]
        end_to_begin = begin - end
        begin_to_end = end - begin

        end_to_query = query - end
        begin_to_query = query - begin

        # Cal the degree between two vectors
        cos_end = np.dot(end_to_begin, end_to_query) / (np.linalg.norm(end_to_begin) * np.linalg.norm(
            end_to_query))

        cos_begin = np.dot(begin_to_end, begin_to_query) / (np.linalg.norm(begin_to_end) * np.linalg.norm(
            begin_to_query))
        # Judge whether the cos >= 90'; if >=90', the nearest point of this line is the vertices of this line
        # because we calcualte it out of this function, so just fill MAX
        if cos_end <= 0 or cos_begin <= 0:
            return 9999999, begin
        else:
            # Calculate the projected distance and point
            sin_end = np.sqrt(1 - cos_end ** 2)
            distance = sin_end * np.linalg.norm(end_to_query)

            # find target point in the line
            end_to_query_mapped_line_value = cos_end * np.linalg.norm(end_to_query)
            end_to_begin_direction = end_to_begin / np.linalg.norm(end_to_begin)
            target_point = end + end_to_query_mapped_line_value * end_to_begin_direction
            return distance, target_point
