import copy
import os
import math
import numpy as np
import openmesh
from enum import IntEnum
from utils.colormap import ColorMapType
from utils.maths import gaussian, random_vector
from geometry.Geometry import Geometry

from tqdm import tqdm
from scipy import optimize, stats




class NoiseDirection(IntEnum):
    NORMAL = 0
    RANDOM = 1


class Mesh(Geometry):

    def __init__(self):
        super().__init__()
        self.render_edges = True
        self.render_flat_faces = False
        self.edge_color = np.array([0.0, 0.0, 0.0, 1.0])
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
            data.add_edges(p1, p2, np.array([self.edge_color]))
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
        verts = np.empty((tri_mesh.n_vertices(), 3))
        faces = np.empty((tri_mesh.n_faces(), 3), dtype=np.uint32)
        f_to_f = np.empty((tri_mesh.n_faces(), 1), dtype=np.uint32)
        norms = np.empty((tri_mesh.n_faces(), 3))
        if mesh.has_vertex_texcoords2D():
            texs = np.empty((tri_mesh.n_vertices(), 2))
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
        edges1 = np.empty((mesh.n_edges(), 3))
        edges2 = np.empty((mesh.n_edges(), 3))
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
        p = np.array([0.0, 0.0, 0.0])
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
        barycenter = [np.array([0.0, 0.0, 0.0])] * self.mesh.n_faces()
        area = [0.0] * self.mesh.n_faces()

        # loop over faces
        for fh in self.mesh.faces():

            # compute barycenter of face
            center = np.array([0.0, 0.0, 0.0])
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
                a = 0.5 * np.linalg.norm(np.cross((v0 - v1), (v2 - v1)))
                area[fh.idx()] = a
                total_area += area[fh.idx()]

            elif valence == 4:
                v0 = self.mesh.point(vertices[0])
                v1 = self.mesh.point(vertices[1])
                v2 = self.mesh.point(vertices[2])
                v3 = self.mesh.point(vertices[3])

                # A = 0.5 * || (v0 - v1) x (v2 - v1) ||
                a012 = np.linalg.norm(np.cross((v0 - v1), (v2 - v1)))
                a023 = np.linalg.norm(np.cross((v0 - v2), (v3 - v2)))
                a013 = np.linalg.norm(np.cross((v0 - v1), (v3 - v1)))
                a123 = np.linalg.norm(np.cross((v1 - v2), (v3 - v2)))
                area[fh.idx()] = (a012 + a023 + a013 + a123) * 0.25
                total_area += area[fh.idx()]

            else:
                print("Error: Arbitrary polygonal faces not supported")
                return

        # compute mesh centroid
        centroid = np.array([0.0, 0.0, 0.0])
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

    # def mesh_parameterization(self):
    #     M_orignal = np.array([self.mesh.point(vh) for vh in self.mesh.vertices()])
    #     num_v = M_orignal.shape[0]

    #     lambda_f0 = np.array([1 for _ in range(len(self.mesh.faces()))])
    #     x0 = np.append(M_orignal[:,:2].flatten('F'), lambda_f0)
    #     w_conf, w_lambda, w_fair = 1, 1, 0.1

    #     def objective(x):
    #         E_total = 0

    #         M_p_x = x[:num_v]
    #         M_p_y = x[num_v:2*num_v]
    #         lambda_f_array = x[2*num_v:]

    #         for fa_idx, fa in enumerate(self.mesh.faces()):
    #             lambda_f = lambda_f_array[fa_idx]
    #             v_o_xyz = np.array([M_orignal[v.idx()] for v in self.mesh.fv(fa)])
    #             v_p_xy = np.array([np.array([M_p_x[v.idx()], M_p_y[v.idx()]]) for v in self.mesh.fv(fa)])

    #             c_conf_0 = lambda_f*np.dot(v_o_xyz[0] - v_o_xyz[2], v_o_xyz[0] - v_o_xyz[2]) - \
    #                 np.dot(v_p_xy[0]-v_p_xy[2], v_p_xy[0]-v_p_xy[2])
    #             c_conf_1 = lambda_f*np.dot(v_o_xyz[1] - v_o_xyz[3], v_o_xyz[1] - v_o_xyz[3]) - \
    #                 np.dot(v_p_xy[1]-v_p_xy[3], v_p_xy[1]-v_p_xy[3])
    #             c_conf_2 = lambda_f*np.dot(v_o_xyz[0] - v_o_xyz[2], v_o_xyz[1] - v_o_xyz[3]) - \
    #                 np.dot(v_p_xy[0]-v_p_xy[2], v_p_xy[1]-v_p_xy[3])
    #             E_conf = w_conf*(c_conf_0**2 + c_conf_1**2 + c_conf_2**2)
    #             E_lambda = w_lambda*(lambda_f - 1)**2

    #             E_total += E_conf + E_lambda

    #         for vh in self.mesh.vertices():
    #             neighbor_vertices = self.mesh.vv(vh)
    #             neighbor_idxs = [i.idx() for i in neighbor_vertices]
    #             if len(neighbor_idxs) == 3:
    #                 v_k_xy = np.array([M_p_x[vh.idx()], M_p_y[vh.idx()]])
    #                 v_i_xy = np.array([M_p_x[neighbor_idxs[0]], M_p_y[neighbor_idxs[0]]])
    #                 v_j_xy = np.array([M_p_x[neighbor_idxs[2]], M_p_y[neighbor_idxs[2]]])
    #                 E_fair = np.dot(v_i_xy - 2*v_k_xy + v_j_xy, v_i_xy - 2*v_k_xy + v_j_xy)

    #                 E_total += w_fair*E_fair

    #         print(E_total)
    #         return E_total
    #     res = optimize.minimize(objective, x0)


    def curvatures(self, k=10):
        gauss = []
        mean = []

        # Step1: find k neighbors of each vertex
        for vh in tqdm(self.mesh.vertices()):
            neighbors = []
            neighborhood = [vh]

            while len(neighbors) < k:
                temp = []
                for v in neighborhood:
                    for neighbor_of_v in self.mesh.vv(v):
                        if neighbor_of_v not in neighbors:
                            neighbors.append(neighbor_of_v)
                            temp.append(neighbor_of_v)
                neighborhood = temp
            neighbors = neighbors[:k]
            point_list = [self.mesh.point(v) for v in neighbors]

        # Step2: convert points into the Cartesian coordinate system
            aligned_point_list = []
            normal_vh = self.mesh.normal(vh)
            transform_matrix = self._transform_matrix(normal_vh, [0, 0, 1])

            for point in point_list:
                p_vec = np.array([point[0], point[1], point[2]])
                aligned_point_list.append(
                    np.dot(transform_matrix, (p_vec - self.mesh.point(vh)))
                )

        # Step3: fit points to z = a_0 + a_1 x + a_2 y + a_3 x^2 + a_4 xy + a_5 y^2
            p = np.vstack(aligned_point_list)
            x, y, z = p[:,0], p[:,1], p[:,2]
            def objective(a):
                f = (a[0] + a[1]*x + a[2]*y + a[3]*x**2 + a[4]*x*y + a[5]*y**2-z)**2
                return np.sum(f)

            res = optimize.minimize(objective, np.zeros((6)))
            gauss_vh = res.x[3] * res.x[5]
            mean_vh = (res.x[3] + res.x[5])/2

            gauss.append(gauss_vh)
            mean.append(mean_vh)

        # Step4: plot of curvature values in the (H, K) plane
        gauss_array = np.array(gauss)
        mean_array = np.array(mean)
        with open("curvature.npy", "wb") as f:
            np.save(f, gauss_array)
            np.save(f, mean_array)
        # with open("mean.npy", "wb") as f:
        return gauss_array.reshape((-1, 1)), mean_array.reshape((-1, 1))


    def asymptotic_directions(self):
        # TODO: compute asymptotic directions
        indices = np.arange(self.mesh.n_vertices())
        normals = self.mesh.vertex_normals()
        return indices, normals


    def _transform_matrix(self, v1, v2):
        """Generates a matrix that transforms v1 into v2"""
        a = (v1/np.linalg.norm(v1)).reshape(3)
        b = (v2/np.linalg.norm(v2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        transform_matrix = np.eye(3) + kmat + kmat.dot(kmat)*((1 - c)/(s**2))
        return transform_matrix


    def write2obj(self, M, filename = 'result.obj'):
        """
        write to obj
        :return:
        """

        f = open(filename, "w+")
        for i in range(self.num_vertices()):
            vinfo = "v " + str(M[i][0]) + " " + str(M[i][1]) + " " + str(M[i][2]) + "\n"
            f.write(vinfo)

        for face_id, face in enumerate(self.mesh.faces()):
            temF = []
            for tem in self.mesh.fv(face):
                temF.append(tem.idx()+1)
            finfo = "f " + str(temF[0]) + " " + str(temF[1]) + " " + str(temF[2]) + " " + str(temF[3]) + "\n"
            f.write(finfo)
        f.close()

    def mesh_parameterization(self):
        # Data setting
        # M = all_points * 3
        points_list = [self.mesh.point(vh) for vh in self.mesh.vertices()]

        # aligned_points_list = []

        # for vh in self.mesh.vertices():
        #     normal_vh = self.mesh.normal(vh)
        #     transform_matrix = self._transform_matrix(normal_vh, [0, 0, 1])
        #     aligned_points_list.append(np.dot(transform_matrix, self.mesh.point(vh)))

        M = np.array(points_list)

        # initial value for M'
        M_prime = np.array(points_list)
        M_prime[:,2] = 0
        # init lambda f
        lambda_f = [1] * len(self.mesh.faces())

        # energy thresholds
        num_iter = 5
        w_conf, w_lambda, w_f = 1, 1, 1

        # Jaccob and f(x)
        num_constraints = 3 * self.mesh.n_faces() + self.mesh.n_faces() + 2 * 3 * self.num_vertices()
        num_variables = 3 * self.num_vertices() + self.mesh.n_faces()
        face_array = self.mesh.face_vertex_indices()

        for iter in range(num_iter):
            print("iteration {}".format(iter))
            J = np.zeros((num_constraints, num_variables))
            f = np.zeros((num_constraints, 1))

            # conf constraints
            for face_id, face in tqdm(enumerate(self.mesh.faces())):
                # face-vertices index and coordinates
                tem_points_index = face_array[face_id]
                tem_origin_points_coor = [points_list[i] for i in tem_points_index]
                tem_new_points_coor = [M_prime[i] for i in tem_points_index]

                # lambda
                tem_lambda = lambda_f[face_id]

                # c_conf,0  var: v0, v2, lambda
                J[3 * face_id][(3 * tem_points_index[0]) : (3 * tem_points_index[0] + 3)] = -2 * (tem_new_points_coor[0] - tem_new_points_coor[2])
                J[3 * face_id][(3 * tem_points_index[2]) : (3 * tem_points_index[2] + 3)] = 2 * (tem_new_points_coor[0] - tem_new_points_coor[2])
                J[3 * face_id][3 * self.num_vertices() + face_id] = np.linalg.norm(tem_origin_points_coor[0] - tem_origin_points_coor[2]) ** 2
                f[3 * face_id] = tem_lambda * np.linalg.norm(tem_origin_points_coor[0] - tem_origin_points_coor[2]) ** 2 \
                                - np.linalg.norm(tem_new_points_coor[0] - tem_new_points_coor[2]) ** 2

                # c_conf,1
                J[3 * face_id + 1][(3 * tem_points_index[1]): (3 * tem_points_index[1] + 3)] = -2 * (tem_new_points_coor[1] - tem_new_points_coor[3])
                J[3 * face_id + 1][(3 * tem_points_index[3]): (3 * tem_points_index[3] + 3)] = 2 * (tem_new_points_coor[1] - tem_new_points_coor[3])
                J[3 * face_id + 1][3 * self.num_vertices() + face_id] = np.linalg.norm(tem_origin_points_coor[1] - tem_origin_points_coor[3]) ** 2
                f[3 * face_id + 1] = tem_lambda * np.linalg.norm(tem_origin_points_coor[1] - tem_origin_points_coor[3]) ** 2 \
                                 - np.linalg.norm(tem_new_points_coor[1] - tem_new_points_coor[3]) ** 2

                # c_conf, 2
                J[3 * face_id + 2][(3 * tem_points_index[0]) : (3 * tem_points_index[0] + 3)] = -1 * (tem_new_points_coor[1] - tem_new_points_coor[3])
                J[3 * face_id + 2][(3 * tem_points_index[1]) : (3 * tem_points_index[1] + 3)] = -1 * (tem_new_points_coor[0] - tem_new_points_coor[2])
                J[3 * face_id + 2][(3 * tem_points_index[2]) : (3 * tem_points_index[2] + 3)] = tem_new_points_coor[1] - tem_new_points_coor[3]
                J[3 * face_id + 2][(3 * tem_points_index[3]) : (3 * tem_points_index[3] + 3)] = tem_new_points_coor[0] - tem_new_points_coor[2]
                J[3 * face_id + 2][3 * self.num_vertices() + face_id] = np.dot((tem_origin_points_coor[0] - tem_origin_points_coor[2]), (tem_origin_points_coor[1] - tem_origin_points_coor[3]))
                f[3 * face_id + 2] = tem_lambda * np.dot((tem_origin_points_coor[0] - tem_origin_points_coor[2]), (tem_origin_points_coor[1] - tem_origin_points_coor[3])) \
                                    - np.dot((tem_new_points_coor[0] - tem_new_points_coor[2]), (tem_new_points_coor[1] - tem_new_points_coor[3]))

            J = J * w_conf
            f = f * w_conf

            # lambda constraints
            for face_id, face in tqdm(enumerate(self.mesh.faces())):
                tem_lambda = lambda_f[face_id]

                J[3 * self.mesh.n_faces() + face_id][3 * self.num_vertices() + face_id] = 1 * w_lambda
                f[3 * self.mesh.n_faces() + face_id] = (tem_lambda - 1) * w_lambda

            # fairness
            cnt_const = 3 * self.mesh.n_faces() + self.mesh.n_faces()
            for idx_vertex, vh in enumerate(self.mesh.vertices()):
                vertex_list = []
                for vertex in self.mesh.vv(vh):
                    vertex_list.append(vertex)
                idx_v_list = [vertex.idx() for vertex in vertex_list]
                point_list = [self.mesh.point(vertex) for vertex in vertex_list]
                point_vh = self.mesh.point(vh)

                if self.mesh.is_boundary(vh):
                    if len(vertex_list) == 3:
                        for i in range(3):
                            J[cnt_const + 6 * idx_vertex + i][3 * idx_v_list[0] + i] = 1 * w_f  # dc/dv0
                            J[cnt_const + 6 * idx_vertex + i][3 * idx_v_list[2] + i] = 1 * w_f  # dc/dv2
                            J[cnt_const + 6 * idx_vertex + i][3 * vh.idx() + i] = -2 * w_f  # dc/dvi
                            f[cnt_const + 6 * idx_vertex + i] = (point_list[0][i] + point_list[2][i] - 2 * point_vh[
                                i]) * w_f
                    continue

                if len(vertex_list) != 4:
                    continue

                for i in range(3):
                    J[cnt_const + 6 * idx_vertex + i][3 * idx_v_list[0] + i] = 1 * w_f  # dc/dv0
                    J[cnt_const + 6 * idx_vertex + i][3 * idx_v_list[2] + i] = 1 * w_f  # dc/dv2
                    J[cnt_const + 6 * idx_vertex + i][3 * vh.idx() + i] = -2 * w_f  # dc/dvi
                    f[cnt_const + 6 * idx_vertex + i] = (point_list[0][i] + point_list[2][i] - 2 * point_vh[i]) * w_f

                    J[cnt_const + 6 * idx_vertex + 3 + i][3 * idx_v_list[1] + i] = 1 * w_f  # dc/dv1
                    J[cnt_const + 6 * idx_vertex + 3 + i][3 * idx_v_list[3] + i] = 1 * w_f  # dc/dv3
                    J[cnt_const + 6 * idx_vertex + 3 + i][3 * vh.idx() + i] = -2 * w_f  # dc/dvi
                    f[cnt_const + 6 * idx_vertex + 3 + i] = (point_list[1][i] + point_list[3][i] - 2 * point_vh[i]) * w_f

            J_trans = np.transpose(J)
            H = np.dot(J_trans, J)
            B = -np.dot(J_trans, f)
            print("start")
            try:
                delta = np.linalg.solve(H, B)   # variables * 1
            except:
                print("singular")
                delta = np.matmul(np.linalg.pinv(H), B)
            print("end")
            # update M prime M_prime
            update_lr = 0.5  # liiquid
            update_lr = 0.5
            for ver_id, vertex in enumerate(self.mesh.vertices()):
                # pdb.set_trace()
                M_prime[ver_id] = M_prime[ver_id] + update_lr * delta[3 * ver_id: 3 * ver_id + 3].flatten()
                # if iter == num_iter - 1:
                    # M_prime[ver_id][2] = 0
                self.mesh.set_point(vertex, M_prime[ver_id])
                    # pdb.set_trace()

            # update lambda
            lambda_f = lambda_f + update_lr * delta[3 * self.num_vertices(): ].flatten()

            # log error
            E_conf = np.dot(np.transpose(f[:3 * self.mesh.n_faces()]), f[:3 * self.mesh.n_faces()])
            E_lambda = np.dot(np.transpose(f[3 * self.mesh.n_faces():]), f[3 * self.mesh.n_faces():])
            print("Iteration {} - E conf: {}".format(iter, E_conf))
            print("Iteration {} - E lambda: {}".format(iter, E_lambda))

        print("Optimization Finish")