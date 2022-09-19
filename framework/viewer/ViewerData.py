import math
import copy
import numpy
import OpenGL.GL
from utils.maths import clamp, normalize
from utils.color import GOLD_AMBIENT, GOLD_DIFFUSE, GOLD_SPECULAR
from utils.colormap import ColorMapType, ColorMap
from viewer.opengl.MeshGL import DirtyFlags, MeshGL


class ViewerData(object):

    def __init__(self):

        self.dirty = DirtyFlags.DIRTY_ALL
        self.meshgl = MeshGL()

        self.V = numpy.empty((0, 3))
        self.F = numpy.empty((0, 3), dtype=numpy.uint32)
        self.FtoF = numpy.empty((0, 1), dtype=numpy.uint32)

        self.V_normals = numpy.empty((0, 3))
        self.F_normals = numpy.empty((0, 3))

        self.V_uv = numpy.empty((0, 2))
        self.F_uv = numpy.empty((0, 3))

        self.F_material_ambient = numpy.empty((0, 4))
        self.F_material_diffuse = numpy.empty((0, 4))
        self.F_material_specular = numpy.empty((0, 4))

        self.V_material_ambient = numpy.empty((0, 4))
        self.V_material_diffuse = numpy.empty((0, 4))
        self.V_material_specular = numpy.empty((0, 4))

        self.texture_R = numpy.empty(0)
        self.texture_G = numpy.empty(0)
        self.texture_B = numpy.empty(0)
        self.texture_A = numpy.empty(0)

        self.lines = numpy.empty((0, 9))
        self.points = numpy.empty((0, 6))

        self.vertex_labels_positions = numpy.empty((0, 3))
        self.face_labels_positions = numpy.empty((0, 3))
        self.labels_positions = numpy.empty((0, 3))
        self.vertex_labels_strings = []
        self.face_labels_strings = []
        self.labels_strings = []

        self.face_based = False
        self.double_sided = False
        self.invert_normals = False
        self.show_texture = False
        self.use_matcap = False

        self.id = -1
        self.is_visible = 1
        self.show_faces = 1
        self.show_lines = 1
        self.show_overlay = 1
        self.show_overlay_depth = 1
        self.show_vertex_labels = 0
        self.show_face_labels = 0
        self.show_custom_labels = 0

        self.line_color = numpy.array([0, 0, 0, 1])
        self.label_color = numpy.array([0, 0, 0.04, 1])
        self.line_width = 0.5
        self.point_size = 30.0
        self.shininess = 35.0

    def set_face_based(self, new_value):
        if self.face_based != new_value:
            self.face_based = new_value
            self.dirty = DirtyFlags.DIRTY_ALL

    def set_mesh(self, v, f, f_to_f=None):
        # If V only has two columns, pad with a column of zeros
        if v.shape[1] == 2:
            v_temp = numpy.zeros((v.shape[0], 3))
            v_temp[:, 0:2] = v
        else:
            v_temp = v

        if self.V.shape[0] == 0 and self.F.shape[0] == 0:
            self.V = copy.deepcopy(v_temp)
            self.F = copy.deepcopy(f)

            self.compute_normals()
            self.uniform_colors(GOLD_AMBIENT, GOLD_DIFFUSE, GOLD_SPECULAR)

        else:
            if v.shape[0] == self.V.shape[0] and f.shape[0] == self.F.shape[0]:
                self.V = copy.deepcopy(v_temp)
                self.F = copy.deepcopy(f)

            else:
                print("ERROR (set_mesh): The new mesh has a different number of vertices/faces. "
                      "Please clear the mesh before plotting.")

        if f_to_f is not None:
            self.FtoF = f_to_f

        self.dirty |= DirtyFlags.DIRTY_FACE | DirtyFlags.DIRTY_POSITION

    def set_vertices(self, v):
        self.V = copy.deepcopy(v)
        assert (self.F.size == 0 or numpy.max(self.F) < self.V.shape[0])
        self.dirty |= DirtyFlags.DIRTY_POSITION

    def set_normals(self, n):

        if n.shape[0] == self.V.shape[0]:
            self.set_face_based(False)
            self.V_normals = n

        elif n.shape[0] == self.F.shape[0] or n.shape[0] == self.F.shape[0] * 3:
            self.set_face_based(True)
            self.F_normals = n

        else:
            print("ERROR (set_normals): Please provide a normal per face, per corner or per vertex.")

        self.dirty |= DirtyFlags.DIRTY_NORMAL

    def set_visible(self, value, core_id=1):

        if value:
            self.is_visible |= core_id
        else:
            self.is_visible &= ~core_id

    def copy_options(self, from_core, to_core):
        self.show_overlay = to_core.set(self.show_overlay, from_core.is_set(self.show_overlay))
        self.show_overlay_depth = to_core.set(self.show_overlay_depth, from_core.is_set(self.show_overlay_depth))
        self.show_texture = to_core.set(self.show_texture, from_core.is_set(self.show_texture))
        self.use_matcap = to_core.set(self.use_matcap, from_core.is_set(self.use_matcap))
        self.show_faces = to_core.set(self.show_faces, from_core.is_set(self.show_faces))
        self.show_lines = to_core.set(self.show_lines, from_core.is_set(self.show_lines))

    @staticmethod
    def ambient(colors):
        # Ambient color should be darker color
        result = 0.1 * colors
        result[:, 3] = colors[:, 3]
        return result

    @staticmethod
    def specular(colors):
        # Specular color should be a less saturated and darker color: dampened highlights
        grey = 0.3
        result = grey + 0.1 * (colors - grey)
        result[:, 3] = colors[:, 3]
        return result

    def set_colors(self, c):

        # This Gouraud coloring should be deprecated in favor of Phong coloring in set_data
        if c.shape[0] > 0 and c.shape[1] == 1:
            assert (False and "deprecated: call set_data directly instead")
            return self.set_data(c)

        if c.shape[0] == 1:

            for i in range(self.V_material_diffuse.shape[0]):

                if c.shape[1] == 3:
                    self.V_material_diffuse[i, :] = numpy.array([c[0, 0], c[0, 1], c[0, 2], 1.0])
                elif c.shape[1] == 4:
                    self.V_material_diffuse[i, :] = numpy.array([c[0, 0], c[0, 1], c[0, 2], c[0, 3]])

            self.V_material_ambient = ViewerData.ambient(self.V_material_diffuse)
            self.V_material_specular = ViewerData.specular(self.V_material_diffuse)

            for i in range(self.F_material_diffuse.shape[0]):

                if c.shape[1] == 3:
                    self.F_material_diffuse[i, :] = numpy.array([c[0, 0], c[0, 1], c[0, 2], 1.0])
                elif c.shape[1] == 4:
                    self.F_material_diffuse[i, :] = numpy.array([c[0, 0], c[0, 1], c[0, 2], c[0, 3]])

            self.F_material_ambient = ViewerData.ambient(self.F_material_diffuse)
            self.F_material_specular = ViewerData.specular(self.F_material_diffuse)

        elif c.shape[0] == self.V.shape[0] or c.shape[0] == self.F.shape[0]:

            # face based colors?
            if c.shape[0] == self.F.shape[0] and (c.shape[0] != self.V.shape[0] or self.face_based):
                self.set_face_based(True)

                for i in range(self.F_material_diffuse.shape[0]):

                    if c.shape[1] == 3:
                        self.F_material_diffuse[i, :] = numpy.array([c[i, 0], c[i, 1], c[i, 2], 1.0])
                    elif c.shape[1] == 4:
                        self.F_material_diffuse[i, :] = numpy.array([c[i, 0], c[i, 1], c[i, 2], c[i, 3]])

                self.F_material_ambient = ViewerData.ambient(self.F_material_diffuse)
                self.F_material_specular = ViewerData.specular(self.F_material_diffuse)

            else:  # c.shape[0] == self.V.shape[0]
                self.set_face_based(False)

                for i in range(self.V_material_diffuse.shape[0]):

                    if c.shape[1] == 3:
                        self.V_material_diffuse[i, :] = numpy.array([c[i, 0], c[i, 1], c[i, 2], 1.0])
                    elif c.shape[1] == 4:
                        self.V_material_diffuse[i, :] = numpy.array([c[i, 0], c[i, 1], c[i, 2], c[i, 3]])

                self.V_material_ambient = ViewerData.ambient(self.V_material_diffuse)
                self.V_material_specular = ViewerData.specular(self.V_material_diffuse)

        else:
            print("ERROR (set_colors): Please provide a single color, or a color per face or per vertex.")

        self.dirty |= DirtyFlags.DIRTY_DIFFUSE | DirtyFlags.DIRTY_SPECULAR | DirtyFlags.DIRTY_AMBIENT

    def set_uv(self, uv_v, uv_f=None):

        if uv_f is None:
            if uv_v.shape[0] == self.V.shape[0]:
                self.set_face_based(False)
                self.V_uv = uv_v
            else:
                print("ERROR (set_UV): Please provide uv per vertex.")

        else:
            self.set_face_based(True)
            self.V_uv = uv_v[:, 0:2]
            self.F_uv = uv_f

        self.dirty |= DirtyFlags.DIRTY_UV

    def set_texture(self, r, g, b, a=None):
        self.texture_R = r
        self.texture_G = g
        self.texture_B = b
        if a is None:
            self.texture_A = 255.0 * numpy.ones(r.shape)
        else:
            self.texture_A = a
        self.dirty |= DirtyFlags.DIRTY_TEXTURE

    def set_data(self, data, c_axis_min=None, c_axis_max=None, c_map=ColorMapType.COLOR_MAP_TYPE_VIRIDIS, num_steps=21):

        if c_axis_min is None:
            c_axis_min = numpy.min(data)

        if c_axis_max is None:
            c_axis_max = numpy.max(data)

        if not self.show_texture:
            colors = ColorMap.colormap(c_map, numpy.array([numpy.linspace(0, 1, num_steps)]).transpose(), 0, 1)
            self.set_colormap(colors)

        tex = (data - c_axis_min) / (c_axis_max - c_axis_min)
        self.set_uv(numpy.repeat(tex, 2, axis=1))

    def set_colormap(self, colormap):
        assert (colormap.shape[1] == 3 and "colormap CM should have 3 columns")

        # Convert to R, G, B textures
        r = colormap[:, 0] * 255.0
        g = colormap[:, 1] * 255.0
        b = colormap[:, 2] * 255.0

        self.set_colors(numpy.array([[1.0, 1.0, 1.0]]))
        self.set_texture(r, g, b)
        self.show_texture = 1
        self.meshgl.tex_filter = OpenGL.GL.GL_NEAREST
        self.meshgl.tex_wrap = OpenGL.GL.GL_CLAMP_TO_EDGE

    def set_points(self, p, c):
        self.points = numpy.empty((0, 0))  # clear existing points
        self.add_points(p, c)

    def add_points(self, p, c):

        # If P only has two columns, pad with a column of zeros
        if p.shape[1] == 2:
            p_temp = numpy.zeros((p.shape[0], 3))
            p_temp[:, 0:2] = p
        else:
            p_temp = p

        last_id = self.points.shape[0]
        self.points = numpy.resize(self.points, (self.points.shape[0] + p_temp.shape[0], 6))
        for i in range(p_temp.shape[0]):
            if i < c.shape[0]:
                j = i
            else:
                j = c.shape[0] - 1
            self.points[last_id + i, :] = numpy.array(
                [p_temp[i, 0], p_temp[i, 1], p_temp[i, 2], c[j, 0], c[j, 1], c[j, 2]])

        self.dirty |= DirtyFlags.DIRTY_OVERLAY_POINTS

    def clear_points(self):
        self.points = numpy.empty((0, 6))

    def set_edges(self, p, e, c):

        self.lines = numpy.empty((e.shape[0], 9))
        assert (c.shape[1] == 3)
        for i in range(e.shape[0]):
            if c.size == 3:
                color = c
            elif c.shape[0] == e.shape[0]:
                color = c[i]
            else:
                assert (False and "Error: set_edges()")
                color = []
            self.lines[i] = numpy.array(
                [p[e[i, 0], 0], p[e[i, 0], 1], p[e[i, 0], 2],
                 p[e[i, 1], 0], p[e[i, 1], 1], p[e[i, 1], 2],
                 color[0], color[1], color[2]])

        self.dirty |= DirtyFlags.DIRTY_OVERLAY_LINES

    def set_edges_from_vector_field(self, p, v, c):
        assert (p.shape[0] == v.shape[0])
        e = numpy.empty((p.shape[0], 2))
        pv = numpy.empty((2 * p.shape[0], 3))
        pv[:p.shape[0], :] = p
        pv[p.shape[0]:, :] = p + v
        for i in range(p.shape[0]):
            e[i, 0] = i
            e[i, 1] = i + p.shape[0]
        self.set_edges(pv, e, c)

    def add_edges(self, p1, p2, c):

        # If P1 only has two columns, pad with a column of zeros
        if p1.shape[1] == 2:
            p1_temp = numpy.zeros((p1.shape[0], 3))
            p1_temp[:, 0:2] = p1
            p2_temp = numpy.zeros((p2.shape[0], 3))
            p2_temp[:, 0:2] = p2
        else:
            p1_temp = p1
            p2_temp = p2

        last_id = self.lines.shape[0]
        self.lines = numpy.resize(self.lines, (self.lines.shape[0] + p1_temp.shape[0], 9))
        for i in range(p1_temp.shape[0]):
            if i < c.shape[0]:
                j = i
            else:
                j = c.shape[0] - 1
            self.lines[last_id + i, :] = numpy.array(
                [p1_temp[i, 0], p1_temp[i, 1], p1_temp[i, 2],
                 p2_temp[i, 0], p2_temp[i, 1], p2_temp[i, 2],
                 c[j, 0], c[j, 1], c[j, 2]])

        self.dirty |= DirtyFlags.DIRTY_OVERLAY_LINES

    def clear_edges(self):
        self.lines = numpy.empty((0, 9))

    def add_label(self, p, string):
        # If P only has two columns, pad with a column of zeros
        if p.size == 2:
            p_temp = numpy.zeros(3)
            p_temp[0:2] = p
        else:
            p_temp = p

        last_id = self.labels_positions.shape[0]
        self.labels_positions = numpy.resize(self.labels_positions, (self.labels_positions.shape[0] + 1, 3))
        self.labels_positions[last_id, :] = p_temp
        self.labels_strings.append(string)

        self.dirty |= DirtyFlags.DIRTY_CUSTOM_LABELS

    def set_labels(self, p, strings):
        assert (p.shape[0] == len(strings) and "position # and label # do not match!")
        assert (p.shape[1] == 3 and "dimension of label positions incorrect!")
        self.labels_positions = p
        self.labels_strings = strings

    def clear_labels(self):
        self.labels_positions = numpy.empty((0, 3))
        self.labels_strings = []

    def clear(self):

        self.V = numpy.empty((0, 3))
        self.F = numpy.empty((0, 3), dtype=numpy.uint32)
        self.FtoF = numpy.empty((0, 1), dtype=numpy.uint32)

        self.V_normals = numpy.empty((0, 3))
        self.F_normals = numpy.empty((0, 3))

        self.V_uv = numpy.empty((0, 2))
        self.F_uv = numpy.empty((0, 3))

        self.F_material_ambient = numpy.empty((0, 4))
        self.F_material_diffuse = numpy.empty((0, 4))
        self.F_material_specular = numpy.empty((0, 4))

        self.V_material_ambient = numpy.empty((0, 4))
        self.V_material_diffuse = numpy.empty((0, 4))
        self.V_material_specular = numpy.empty((0, 4))

        self.lines = numpy.empty((0, 9))
        self.points = numpy.empty((0, 6))

        self.vertex_labels_positions = numpy.empty((0, 3))
        self.face_labels_positions = numpy.empty((0, 3))
        self.labels_positions = numpy.empty((0, 3))
        self.vertex_labels_strings = []
        self.face_labels_strings = []
        self.labels_strings = []

        self.face_based = False
        self.double_sided = False
        self.invert_normals = False
        self.show_texture = False
        self.use_matcap = False

    @staticmethod
    def proj_double_area(verts, faces, x, y, f):
        rx = verts[faces[f, 0], x] - verts[faces[f, 2], x]
        sx = verts[faces[f, 1], x] - verts[faces[f, 2], x]
        ry = verts[faces[f, 0], y] - verts[faces[f, 2], y]
        sy = verts[faces[f, 1], y] - verts[faces[f, 2], y]
        return rx * sy - ry * sx

    @staticmethod
    def per_vertex_normals(verts, faces, face_normals):

        assert (faces.shape[1] == 3)  # Only support triangles
        double_areas = numpy.zeros((faces.shape[0], 1))
        if verts.shape[1] == 3:
            for f in range(faces.shape[0]):
                for d in range(3):
                    dbl_area = ViewerData.proj_double_area(verts, faces, d, (d + 1) % 3, f)
                    double_areas[f, 0] += dbl_area * dbl_area
            double_areas = numpy.sqrt(double_areas)
        elif verts.shape[1] == 2:
            for f in range(faces.shape[0]):
                double_areas[f, 0] = ViewerData.proj_double_area(verts, faces, 0, 1, f)
        weights = numpy.repeat(double_areas, 3, axis=1)

        # loop over faces
        normals = numpy.zeros((verts.shape[0], 3))
        for i in range(faces.shape[0]):
            # throw normal at each corner
            for j in range(3):
                normals[faces[i, j], :] += weights[i, j] * face_normals[i, :]

        norms = numpy.linalg.norm(normals, axis=1)
        normals = normals / norms[:, numpy.newaxis]
        return normals

    @staticmethod
    def per_face_normals(verts, faces):

        # loop over faces
        normals = numpy.zeros((faces.shape[0], 3))
        for i in range(faces.shape[0]):

            v1 = verts[faces[i, 1], :] - verts[faces[i, 0], :]
            v2 = verts[faces[i, 2], :] - verts[faces[i, 0], :]
            normals[i, :] = numpy.cross(v1, v2)
            r = numpy.linalg.norm(normals[i, :])
            if r == 0.0:
                normals[i, :] = numpy.array([0.0, 0.0, 0.0])
            else:
                normals[i, :] /= r

        return normals

    def compute_normals(self):
        self.F_normals = ViewerData.per_face_normals(self.V, self.F)
        self.V_normals = ViewerData.per_vertex_normals(self.V, self.F, self.F_normals)
        self.dirty |= DirtyFlags.DIRTY_NORMAL

    def uniform_colors(self, ambient, diffuse, specular):

        self.V_material_ambient = numpy.empty((self.V.shape[0], 4))
        self.V_material_diffuse = numpy.empty((self.V.shape[0], 4))
        self.V_material_specular = numpy.empty((self.V.shape[0], 4))

        for i in range(self.V.shape[0]):
            self.V_material_ambient[i, :] = ambient
            self.V_material_diffuse[i, :] = diffuse
            self.V_material_specular[i, :] = specular

        self.F_material_ambient = numpy.empty((self.F.shape[0], 4))
        self.F_material_diffuse = numpy.empty((self.F.shape[0], 4))
        self.F_material_specular = numpy.empty((self.F.shape[0], 4))

        for i in range(self.F.shape[0]):
            self.F_material_ambient[i, :] = ambient
            self.F_material_diffuse[i, :] = diffuse
            self.F_material_specular[i, :] = specular

        self.dirty |= DirtyFlags.DIRTY_SPECULAR | DirtyFlags.DIRTY_DIFFUSE | DirtyFlags.DIRTY_AMBIENT

    def normal_matcap(self):
        size = 512
        self.texture_R = numpy.empty((size, size))
        self.texture_G = numpy.empty((size, size))
        self.texture_B = numpy.empty((size, size))
        for i in range(size):
            x = (float(i) / float(size - 1) * 2.0 - 1.0)
            for j in range(size):
                y = (float(j) / float(size - 1) * 2.0 - 1.0)
                z = math.sqrt(1.0 - min(x * x + y * y, 1.0))
                c = numpy.array([x * 0.5 + 0.5, y * 0.5 + 0.5, z])
                self.texture_R[i, j] = clamp(c[0]) * 255
                self.texture_G[i, j] = clamp(c[1]) * 255
                self.texture_B[i, j] = clamp(c[2]) * 255
        self.texture_A = 255.0 * numpy.ones((size, size))
        self.dirty |= DirtyFlags.DIRTY_TEXTURE

    def empty_texture(self, size):
        self.texture_R = 255.0 * numpy.ones((size, size))
        self.texture_G = 255.0 * numpy.ones((size, size))
        self.texture_B = 255.0 * numpy.ones((size, size))
        self.texture_A = 255.0 * numpy.ones((size, size))
        self.dirty |= DirtyFlags.DIRTY_TEXTURE

    def grid_texture(self):
        size = 128
        size2 = size / 2
        self.texture_R = numpy.empty((size, size))
        for i in range(size):
            for j in range(size):
                self.texture_R[i, j] = 0
                if (i < size2 and j < size2) or (i >= size2 and j >= size2):
                    self.texture_R[i, j] = 255

        self.texture_G = copy.deepcopy(self.texture_R)
        self.texture_B = copy.deepcopy(self.texture_R)
        self.texture_A = 255.0 * numpy.ones((size, size))
        self.dirty |= DirtyFlags.DIRTY_TEXTURE

    def clear_texture(self):
        x_size = self.texture_R.shape[0]
        y_size = self.texture_R.shape[1]
        self.texture_R = 255.0 * numpy.ones((x_size, y_size))
        self.texture_G = 255.0 * numpy.ones((x_size, y_size))
        self.texture_B = 255.0 * numpy.ones((x_size, y_size))
        self.texture_A = 255.0 * numpy.ones((x_size, y_size))
        self.dirty |= DirtyFlags.DIRTY_TEXTURE

    @staticmethod
    def update_labels(labels, positions, strings):
        if positions.shape[0] > 0:

            assert (len(strings) == positions.shape[0])
            num_chars_to_render = 0
            for p in range(positions.shape[0]):
                num_chars_to_render += len(strings[p])
            labels.label_pos_vbo = numpy.empty((num_chars_to_render, 3), dtype=numpy.float32)
            labels.label_char_vbo = numpy.empty((num_chars_to_render, 1), dtype=numpy.float32)
            labels.label_offset_vbo = numpy.empty((num_chars_to_render, 1), dtype=numpy.float32)
            labels.label_indices_vbo = numpy.empty((num_chars_to_render, 1), dtype=numpy.uint32)
            idx = 0
            for s in range(len(strings)):
                label = strings[s]
                for c in range(len(label)):
                    labels.label_pos_vbo[idx, :] = positions[s, :]
                    labels.label_char_vbo[idx] = float(label[c])
                    labels.label_offset_vbo[idx] = c
                    labels.label_indices_vbo[idx] = idx
                    idx += 1

    @staticmethod
    def per_face(data, x):
        assert (x.shape[1] == 4)
        vbo = numpy.empty((data.F.shape[0] * 3, x.shape[1]), dtype=numpy.float32)
        for i in range(data.F.shape[0]):
            for j in range(3):
                vbo[i * 3 + j, :] = x[i, :]
        return vbo

    @staticmethod
    def per_corner(data, x):
        vbo = numpy.empty((data.F.shape[0] * 3, x.shape[1]), dtype=numpy.float32)
        for i in range(data.F.shape[0]):
            for j in range(3):
                vbo[i * 3 + j, :] = x[data.F[i, j], :]
        return vbo

    def update_gl(self, data, invert_normals, meshgl):

        if not meshgl.is_initialized:
            meshgl.init()

        per_corner_uv = (data.F_uv.shape[0] == data.F.shape[0])
        per_corner_normals = (data.F_normals.shape[0] == 3 * data.F.shape[0])
        meshgl.dirty |= data.dirty

        if not data.face_based:

            if not (per_corner_uv or per_corner_normals):

                # Vertex positions
                if meshgl.dirty & DirtyFlags.DIRTY_POSITION:
                    meshgl.V_vbo = numpy.array(data.V, dtype=numpy.float32)

                # Per-vertex material settings
                if meshgl.dirty & DirtyFlags.DIRTY_AMBIENT:
                    meshgl.V_ambient_vbo = numpy.array(data.V_material_ambient, dtype=numpy.float32)
                if meshgl.dirty & DirtyFlags.DIRTY_DIFFUSE:
                    meshgl.V_diffuse_vbo = numpy.array(data.V_material_diffuse, dtype=numpy.float32)
                if meshgl.dirty & DirtyFlags.DIRTY_SPECULAR:
                    meshgl.V_specular_vbo = numpy.array(data.V_material_specular, dtype=numpy.float32)

                # Vertex normals
                if meshgl.dirty & DirtyFlags.DIRTY_NORMAL:
                    meshgl.V_normals_vbo = numpy.array(data.V_normals, dtype=numpy.float32)
                    if invert_normals:
                        meshgl.V_normals_vbo = -meshgl.V_normals_vbo

                # Face indices
                if meshgl.dirty & DirtyFlags.DIRTY_FACE:
                    meshgl.F_vbo = numpy.array(data.F, dtype=numpy.uint32)

                # Texture coordinates
                if meshgl.dirty & DirtyFlags.DIRTY_UV:
                    meshgl.V_uv_vbo = numpy.array(data.V_uv, dtype=numpy.float32)

            else:  # Per vertex properties with per corner UVs

                # Vertex positions
                if meshgl.dirty & DirtyFlags.DIRTY_POSITION:
                    meshgl.V_vbo = ViewerData.per_corner(data, data.V)

                # Per-vertex material settings
                if meshgl.dirty & DirtyFlags.DIRTY_AMBIENT:
                    meshgl.V_ambient_vbo = numpy.empty((data.F.shape[0] * 3, 4), dtype=numpy.float32)
                    for i in range(data.F.shape[0]):
                        for j in range(3):
                            meshgl.V_ambient_vbo[i * 3 + j, :] = data.V_material_ambient[data.F[i, j], :]

                if meshgl.dirty & DirtyFlags.DIRTY_DIFFUSE:
                    meshgl.V_diffuse_vbo = numpy.empty((data.F.shape[0] * 3, 4), dtype=numpy.float32)
                    for i in range(data.F.shape[0]):
                        for j in range(3):
                            meshgl.V_diffuse_vbo[i * 3 + j, :] = data.V_material_diffuse[data.F[i, j], :]

                if meshgl.dirty & DirtyFlags.DIRTY_SPECULAR:
                    meshgl.V_specular_vbo = numpy.empty((data.F.shape[0] * 3, 4), dtype=numpy.float32)
                    for i in range(data.F.shape[0]):
                        for j in range(3):
                            meshgl.V_specular_vbo[i * 3 + j, :] = data.V_material_specular[data.F[i, j], :]

                # Vertex normals
                if meshgl.dirty & DirtyFlags.DIRTY_NORMAL:
                    meshgl.V_normals_vbo = numpy.empty((data.F.shape[0] * 3, 3), dtype=numpy.float32)
                    for i in range(data.F.shape[0]):
                        for j in range(3):
                            if per_corner_normals:
                                meshgl.V_normals_vbo[i * 3 + j, :] = \
                                    data.F_normals[i * 3 + j, :]
                            else:
                                meshgl.V_normals_vbo[i * 3 + j, :] = \
                                    data.V_normals[data.F[i, j], :]

                    if invert_normals:
                        meshgl.V_normals_vbo = -meshgl.V_normals_vbo

                # Face indices
                if meshgl.dirty & DirtyFlags.DIRTY_FACE:
                    meshgl.F_vbo = numpy.empty((data.F.shape[0], 3), dtype=numpy.uint32)
                    for i in range(data.F.shape[0]):
                        meshgl.F_vbo[i, :] = numpy.array([i * 3 + 0, i * 3 + 1, i * 3 + 2])

                # Texture coordinates
                if meshgl.dirty & DirtyFlags.DIRTY_UV and data.V_uv.shape[0] > 0:
                    meshgl.V_uv_vbo = numpy.empty((data.F.shape[0] * 3, 2), dtype=numpy.float32)
                    for i in range(data.F.shape[0]):
                        for j in range(3):
                            if per_corner_uv:
                                meshgl.V_uv_vbo[i * 3 + j, :] = \
                                    data.V_uv[data.F_uv[i, j], :]
                            else:
                                meshgl.V_uv_vbo[i * 3 + j, :] = \
                                    data.V_uv[data.F[i, j], :]

        else:

            # Vertex positions
            if meshgl.dirty & DirtyFlags.DIRTY_POSITION:
                meshgl.V_vbo = ViewerData.per_corner(data, data.V)

            # Per-vertex material settings
            if meshgl.dirty & DirtyFlags.DIRTY_AMBIENT:
                meshgl.V_ambient_vbo = ViewerData.per_face(data, data.F_material_ambient)
            if meshgl.dirty & DirtyFlags.DIRTY_DIFFUSE:
                meshgl.V_diffuse_vbo = ViewerData.per_face(data, data.F_material_diffuse)
            if meshgl.dirty & DirtyFlags.DIRTY_SPECULAR:
                meshgl.V_specular_vbo = ViewerData.per_face(data, data.F_material_specular)

            # Vertex normals
            if meshgl.dirty & DirtyFlags.DIRTY_NORMAL:
                meshgl.V_normals_vbo = numpy.empty((data.F.shape[0] * 3, 3), dtype=numpy.float32)
                for i in range(data.F.shape[0]):
                    for j in range(3):
                        if per_corner_normals:
                            meshgl.V_normals_vbo[i * 3 + j, :] = \
                                data.F_normals[i * 3 + j, :]
                        else:
                            meshgl.V_normals_vbo[i * 3 + j, :] = \
                                data.F_normals[i, :]

                if invert_normals:
                    meshgl.V_normals_vbo = -meshgl.V_normals_vbo

            # Face indices
            if meshgl.dirty & DirtyFlags.DIRTY_FACE:
                meshgl.F_vbo = numpy.empty((data.F.shape[0], 3), dtype=numpy.uint32)
                for i in range(data.F.shape[0]):
                    meshgl.F_vbo[i, :] = numpy.array([i * 3 + 0, i * 3 + 1, i * 3 + 2])

            # Texture coordinates
            if meshgl.dirty & DirtyFlags.DIRTY_UV and data.V_uv.shape[0] > 0:
                meshgl.V_uv_vbo = numpy.empty((data.F.shape[0] * 3, 2), dtype=numpy.float32)
                for i in range(data.F.shape[0]):
                    for j in range(3):
                        if per_corner_uv:
                            meshgl.V_uv_vbo[i * 3 + j, :] = \
                                data.V_uv[data.F_uv[i, j], :]
                        else:
                            meshgl.V_uv_vbo[i * 3 + j, :] = \
                                data.V_uv[data.F[i, j], :]

        # Texture
        if meshgl.dirty & DirtyFlags.DIRTY_TEXTURE:
            if data.texture_R.ndim == 2:
                meshgl.tex_u = data.texture_R.shape[0]
                meshgl.tex_v = data.texture_R.shape[1]
            elif data.texture_R.ndim == 1:
                meshgl.tex_u = data.texture_R.shape[0]
                meshgl.tex_v = 1
            meshgl.tex = numpy.empty((data.texture_R.size * 4,), dtype=numpy.uint8)
            for i in range(data.texture_R.size):
                meshgl.tex[i * 4 + 0] = data.texture_R[i]
                meshgl.tex[i * 4 + 1] = data.texture_G[i]
                meshgl.tex[i * 4 + 2] = data.texture_B[i]
                meshgl.tex[i * 4 + 3] = data.texture_A[i]

        # Overlay lines
        if meshgl.dirty & DirtyFlags.DIRTY_OVERLAY_LINES:
            meshgl.lines_V_vbo = numpy.empty((data.lines.shape[0] * 2, 3), dtype=numpy.float32)
            meshgl.lines_V_colors_vbo = numpy.empty((data.lines.shape[0] * 2, 3), dtype=numpy.float32)
            meshgl.lines_F_vbo = numpy.empty((data.lines.shape[0] * 2, 1), dtype=numpy.uint32)
            for i in range(data.lines.shape[0]):
                meshgl.lines_V_vbo[2 * i + 0, :] = data.lines[i, 0:3]
                meshgl.lines_V_vbo[2 * i + 1, :] = data.lines[i, 3:6]
                meshgl.lines_V_colors_vbo[2 * i + 0, :] = data.lines[i, 6:9]
                meshgl.lines_V_colors_vbo[2 * i + 1, :] = data.lines[i, 6:9]
                meshgl.lines_F_vbo[2 * i + 0, 0] = 2 * i + 0
                meshgl.lines_F_vbo[2 * i + 1, 0] = 2 * i + 1

        # Overlay points
        if meshgl.dirty & DirtyFlags.DIRTY_OVERLAY_POINTS:
            meshgl.points_V_vbo = numpy.empty((data.points.shape[0], 3), dtype=numpy.float32)
            meshgl.points_V_colors_vbo = numpy.empty((data.points.shape[0], 3), dtype=numpy.float32)
            meshgl.points_F_vbo = numpy.empty((data.points.shape[0], 1), dtype=numpy.uint32)
            for i in range(data.points.shape[0]):
                meshgl.points_V_vbo[i, :] = data.points[i, 0:3]
                meshgl.points_V_colors_vbo[i, :] = data.points[i, 3:6]
                meshgl.points_F_vbo[i, 0] = i

        # Face labels
        if meshgl.dirty & DirtyFlags.DIRTY_FACE_LABELS:
            if self.face_labels_positions.shape[0] == 0:
                self.face_labels_positions = numpy.empty((self.F.shape[0], 3))
                for f in range(self.F.shape[0]):
                    face_normal = normalize(self.F_normals[f, :])
                    face_name = str(f)
                    self.face_labels_positions[f, :] = self.V[self.F[f, 0], :]
                    self.face_labels_positions[f, :] += self.V[self.F[f, 1], :]
                    self.face_labels_positions[f, :] += self.V[self.F[f, 2], :]
                    self.face_labels_positions[f, :] /= 3.0
                    self.face_labels_positions[f, :] += face_normal * 0.05
                    self.face_labels_strings.append(face_name)
            ViewerData.update_labels(meshgl.face_labels, self.face_labels_positions, self.face_labels_strings)

        # Vertex labels
        if meshgl.dirty & DirtyFlags.DIRTY_VERTEX_LABELS:
            if self.vertex_labels_positions.shape[0] == 0:
                self.vertex_labels_positions = numpy.empty((self.V.shape[0], 3))
                for v in range(self.V.shape[0]):
                    vertex_normal = normalize(self.V_normals[v, :])
                    vert_name = str(v)
                    self.vertex_labels_positions[v, :] = self.V[v, :]
                    self.vertex_labels_positions[v, :] += vertex_normal * 0.1
                    self.vertex_labels_strings.append(vert_name)
            ViewerData.update_labels(meshgl.vertex_labels, self.vertex_labels_positions, self.vertex_labels_strings)

        # Custom labels
        if meshgl.dirty & DirtyFlags.DIRTY_CUSTOM_LABELS:
            ViewerData.update_labels(meshgl.custom_labels, self.labels_positions, self.labels_strings)
