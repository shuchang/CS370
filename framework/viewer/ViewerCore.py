import math
import numpy
import OpenGL.GL
import quaternion
from enum import IntEnum
from utils.camera import look_at, ortho, frustum
from utils.camera import snap_to_fixed_up
from viewer.opengl.MeshGL import DirtyFlags


class RotationType(IntEnum):
    ROTATION_TYPE_TRACKBALL = 0
    ROTATION_TYPE_TWO_AXIS_VALUATOR_FIXED_UP = 1
    ROTATION_TYPE_NO_ROTATION = 2
    NUM_ROTATION_TYPES = 3


class ViewerCore(object):

    def __init__(self):

        self.is_visible = True
        # Unique identifier
        self.id = 1

        # Default colors
        self.background_color = numpy.array([0.3, 0.3, 0.5, 1.0])

        # Default lights settings
        self.light_position = numpy.array([0.0, 0.3, 0.0])
        self.lighting_factor = 1.0  # on

        # Default trackball
        self.trackball_angle = quaternion.from_rotation_matrix(numpy.identity(3))
        self.rotation_type = RotationType.ROTATION_TYPE_TRACKBALL
        self.set_rotation_type(RotationType.ROTATION_TYPE_TWO_AXIS_VALUATOR_FIXED_UP)

        # Camera parameters
        self.camera_base_zoom = 1.0
        self.camera_zoom = 1.0
        self.orthographic = False
        self.camera_view_angle = 45.0
        self.camera_dnear = 1.0
        self.camera_dfar = 100.0
        self.camera_base_translation = numpy.array([0, 0, 0])
        self.camera_translation = numpy.array([0, 0, 0])
        self.camera_eye = numpy.array([0, 0, 5])
        self.camera_center = numpy.array([0, 0, 0])
        self.camera_up = numpy.array([0, 1, 0])

        self.depth_test = True

        # Animation
        self.is_animating = False
        self.animation_max_fps = 30.0

        # Caches the two-norm between the min/max point of the bounding box
        self.object_scale = 1.0

        # Viewport size
        self.viewport = numpy.array([0.0, 0.0, 0.0, 0.0])

        # Save the OpenGL transformation matrices used for the previous rendering pass
        self.view = numpy.identity(4)
        self.proj = numpy.identity(4)
        self.norm = numpy.identity(4)

    def init(self):
        pass

    def shut(self):
        pass

    def align_camera_center(self, v, f=None):
        if v.shape[0] == 0:
            return

        self.camera_base_zoom, self.camera_base_translation = \
            ViewerCore.get_scale_and_shift_to_fit_mesh(v, f)

        # Rather than crash on empty mesh...
        if v.size > 0:
            self.object_scale = numpy.linalg.norm(numpy.max(v, axis=0) - numpy.min(v, axis=0))

    @staticmethod
    def get_scale_and_shift_to_fit_mesh(v, f):
        if v.shape[0] == 0:
            return

        if f is None:
            bc = v
        else:
            if f.shape[0] <= 1:
                bc = v
            else:
                bc = numpy.zeros((f.shape[0], v.shape[1]))
                # Loop over faces
                for i in range(f.shape[0]):
                    # loop around face
                    for j in range(f.shape[1]):
                        # Accumulate
                        bc[i, :] += v[f[i, j], :]
                    # average
                    bc[i, :] /= float(f.shape[1])

        min_point = numpy.min(bc, axis=0)
        max_point = numpy.max(bc, axis=0)
        centroid = 0.5 * (min_point + max_point)
        shift = -centroid
        zoom = 2.0 / numpy.max(numpy.abs(max_point - min_point))
        return zoom, shift

    def clear_framebuffers(self):
        # The glScissor call ensures we only clear this core's buffers,
        # (in case the user wants different background colors in each viewport.)
        OpenGL.GL.glScissor(
            self.viewport[0],
            self.viewport[1],
            self.viewport[2],
            self.viewport[3])
        OpenGL.GL.glEnable(OpenGL.GL.GL_SCISSOR_TEST)
        OpenGL.GL.glClearColor(
            self.background_color[0],
            self.background_color[1],
            self.background_color[2],
            self.background_color[3])
        OpenGL.GL.glClear(OpenGL.GL.GL_COLOR_BUFFER_BIT | OpenGL.GL.GL_DEPTH_BUFFER_BIT)
        OpenGL.GL.glDisable(OpenGL.GL.GL_SCISSOR_TEST)

    def draw(self, data, update_matrices=True):

        if self.depth_test:
            OpenGL.GL.glEnable(OpenGL.GL.GL_DEPTH_TEST)
        else:
            OpenGL.GL.glDisable(OpenGL.GL.GL_DEPTH_TEST)

        OpenGL.GL.glEnable(OpenGL.GL.GL_BLEND)
        OpenGL.GL.glBlendFunc(OpenGL.GL.GL_SRC_ALPHA, OpenGL.GL.GL_ONE_MINUS_SRC_ALPHA)

        # Bind and potentially refresh mesh / line / point data
        if data.dirty:
            data.update_gl(data, data.invert_normals, data.meshgl)
            data.dirty = DirtyFlags.DIRTY_NONE
        data.meshgl.bind_mesh()

        # Initialize uniform
        OpenGL.GL.glViewport(
            self.viewport[0],
            self.viewport[1],
            self.viewport[2],
            self.viewport[3])

        if update_matrices:
            self.view = numpy.identity(4)
            self.proj = numpy.identity(4)
            self.norm = numpy.identity(4)

            width = self.viewport[2]
            height = self.viewport[3]

            # Set view
            rotation = numpy.identity(4)
            rotation[0:3, 0:3] = quaternion.as_rotation_matrix(self.trackball_angle)
            scale = numpy.identity(4) * self.camera_zoom * self.camera_base_zoom
            scale[3, 3] = 1.0
            translation = numpy.identity(4)
            translation[0:3, 3] = self.camera_translation + self.camera_base_translation
            self.view = look_at(self.camera_eye, self.camera_center, self.camera_up)
            self.view = self.view @ rotation @ scale @ translation

            self.norm = numpy.transpose(numpy.linalg.inv(self.view))

            # Set projection
            if self.orthographic:
                length = numpy.linalg.norm(self.camera_eye - self.camera_center)
                h = math.tan(self.camera_view_angle / 360.0 * math.pi) * length
                self.proj = ortho(-h * width / height, h * width / height, -h, h,
                                  self.camera_dnear, self.camera_dfar)

            else:
                f_h = math.tan(self.camera_view_angle / 360.0 * math.pi) * self.camera_dnear
                f_w = f_h * float(width) / float(height)
                self.proj = frustum(-f_w, f_w, -f_h, f_h,
                                    self.camera_dnear, self.camera_dfar)

            self.view = self.view.transpose()
            self.norm = self.norm.transpose()
            self.proj = self.proj.transpose()

        # Send transformations to the GPU
        viewi = OpenGL.GL.glGetUniformLocation(data.meshgl.shader_mesh, "view")
        proji = OpenGL.GL.glGetUniformLocation(data.meshgl.shader_mesh, "proj")
        normi = OpenGL.GL.glGetUniformLocation(data.meshgl.shader_mesh, "normal_matrix")
        OpenGL.GL.glUniformMatrix4fv(viewi, 1, OpenGL.GL.GL_FALSE, numpy.array(self.view, dtype=numpy.float32))
        OpenGL.GL.glUniformMatrix4fv(proji, 1, OpenGL.GL.GL_FALSE, numpy.array(self.proj, dtype=numpy.float32))
        OpenGL.GL.glUniformMatrix4fv(normi, 1, OpenGL.GL.GL_FALSE, numpy.array(self.norm, dtype=numpy.float32))

        # Light parameters
        specular_exponenti = OpenGL.GL.glGetUniformLocation(data.meshgl.shader_mesh, "specular_exponent")
        light_position_eyei = OpenGL.GL.glGetUniformLocation(data.meshgl.shader_mesh, "light_position_eye")
        lighting_factori = OpenGL.GL.glGetUniformLocation(data.meshgl.shader_mesh, "lighting_factor")
        fixed_colori = OpenGL.GL.glGetUniformLocation(data.meshgl.shader_mesh, "fixed_color")
        texture_factori = OpenGL.GL.glGetUniformLocation(data.meshgl.shader_mesh, "texture_factor")
        matcap_factori = OpenGL.GL.glGetUniformLocation(data.meshgl.shader_mesh, "matcap_factor")
        double_sidedi = OpenGL.GL.glGetUniformLocation(data.meshgl.shader_mesh, "double_sided")

        OpenGL.GL.glUniform1f(specular_exponenti, data.shininess)
        OpenGL.GL.glUniform3fv(light_position_eyei, 1, numpy.array(self.light_position, dtype=numpy.float32))
        OpenGL.GL.glUniform1f(lighting_factori, self.lighting_factor)  # enables lighting
        OpenGL.GL.glUniform4f(fixed_colori, 0.0, 0.0, 0.0, 0.0)

        if data.V.shape[0] > 0:

            # Render fill
            if self.is_set(data.show_faces):

                # Texture
                if self.is_set(data.show_texture):
                    OpenGL.GL.glUniform1f(texture_factori, 1.0)
                else:
                    OpenGL.GL.glUniform1f(texture_factori, 0.0)

                if self.is_set(data.use_matcap):
                    OpenGL.GL.glUniform1f(matcap_factori, 1.0)
                else:
                    OpenGL.GL.glUniform1f(matcap_factori, 0.0)

                if self.is_set(data.double_sided):
                    OpenGL.GL.glUniform1f(double_sidedi, 1.0)
                else:
                    OpenGL.GL.glUniform1f(double_sidedi, 0.0)

                data.meshgl.draw_mesh(True)
                OpenGL.GL.glUniform1f(matcap_factori, 0.0)
                OpenGL.GL.glUniform1f(texture_factori, 0.0)

            # Render wireframe
            if self.is_set(data.show_lines):
                OpenGL.GL.glLineWidth(data.line_width)
                OpenGL.GL.glUniform4f(
                    fixed_colori,
                    data.line_color[0],
                    data.line_color[1],
                    data.line_color[2],
                    1.0)
                data.meshgl.draw_mesh(False)
                OpenGL.GL.glUniform4f(fixed_colori, 0.0, 0.0, 0.0, 0.0)

        if self.is_set(data.show_overlay):
            if self.is_set(data.show_overlay_depth):
                OpenGL.GL.glEnable(OpenGL.GL.GL_DEPTH_TEST)
            else:
                OpenGL.GL.glDisable(OpenGL.GL.GL_DEPTH_TEST)

            if data.lines.shape[0] > 0:
                data.meshgl.bind_overlay_lines()
                viewi = OpenGL.GL.glGetUniformLocation(data.meshgl.shader_overlay_lines, "view")
                proji = OpenGL.GL.glGetUniformLocation(data.meshgl.shader_overlay_lines, "proj")

                OpenGL.GL.glUniformMatrix4fv(viewi, 1, OpenGL.GL.GL_FALSE, numpy.array(self.view, dtype=numpy.float32))
                OpenGL.GL.glUniformMatrix4fv(proji, 1, OpenGL.GL.GL_FALSE, numpy.array(self.proj, dtype=numpy.float32))
                # This must be enabled, otherwise glLineWidth has no effect
                OpenGL.GL.glEnable(OpenGL.GL.GL_LINE_SMOOTH)
                OpenGL.GL.glLineWidth(data.line_width)

                data.meshgl.draw_overlay_lines()

            if data.points.shape[0] > 0:
                data.meshgl.bind_overlay_points()
                viewi = OpenGL.GL.glGetUniformLocation(data.meshgl.shader_overlay_points, "view")
                proji = OpenGL.GL.glGetUniformLocation(data.meshgl.shader_overlay_points, "proj")

                OpenGL.GL.glUniformMatrix4fv(viewi, 1, OpenGL.GL.GL_FALSE, numpy.array(self.view, dtype=numpy.float32))
                OpenGL.GL.glUniformMatrix4fv(proji, 1, OpenGL.GL.GL_FALSE, numpy.array(self.proj, dtype=numpy.float32))
                OpenGL.GL.glPointSize(data.point_size)
                data.meshgl.draw_overlay_points()

            OpenGL.GL.glEnable(OpenGL.GL.GL_DEPTH_TEST)

        if self.is_set(data.show_vertex_labels) and data.vertex_labels_positions.shape[0] > 0:
            self.draw_labels(data, data.meshgl.vertex_labels)

        if self.is_set(data.show_face_labels) and data.face_labels_positions.shape[0] > 0:
            self.draw_labels(data, data.meshgl.face_labels)

        if self.is_set(data.show_custom_labels) and data.labels_positions.shape[0] > 0:
            self.draw_labels(data, data.meshgl.custom_labels)

    def draw_buffer(self, data, update_matrices, width, height):

        # https://learnopengl.com/Advanced-OpenGL/Anti-Aliasing
        framebuffer = OpenGL.GL.glGenFramebuffers(1)
        OpenGL.GL.glBindFramebuffer(OpenGL.GL.GL_FRAMEBUFFER, framebuffer)
        # create a multisampled color attachment texture
        texture_color_buffer_multi_sampled = OpenGL.GL.glGenTextures(1)
        OpenGL.GL.glBindTexture(OpenGL.GL.GL_TEXTURE_2D_MULTISAMPLE, texture_color_buffer_multi_sampled)
        OpenGL.GL.glTexImage2DMultisample(
            OpenGL.GL.GL_TEXTURE_2D_MULTISAMPLE, 4, OpenGL.GL.GL_RGBA, width, height, OpenGL.GL.GL_TRUE)
        OpenGL.GL.glBindTexture(OpenGL.GL.GL_TEXTURE_2D_MULTISAMPLE, 0)
        OpenGL.GL.glFramebufferTexture2D(
            OpenGL.GL.GL_FRAMEBUFFER, OpenGL.GL.GL_COLOR_ATTACHMENT0, OpenGL.GL.GL_TEXTURE_2D_MULTISAMPLE,
            texture_color_buffer_multi_sampled, 0)
        # create a (also multisampled) renderbuffer object for depth and stencil attachments
        rbo = OpenGL.GL.glGenRenderbuffers(1)
        OpenGL.GL.glBindRenderbuffer(OpenGL.GL.GL_RENDERBUFFER, rbo)
        OpenGL.GL.glRenderbufferStorageMultisample(
            OpenGL.GL.GL_RENDERBUFFER, 4, OpenGL.GL.GL_DEPTH24_STENCIL8, width, height)
        OpenGL.GL.glBindRenderbuffer(OpenGL.GL.GL_RENDERBUFFER, 0)
        OpenGL.GL.glFramebufferRenderbuffer(
            OpenGL.GL.GL_FRAMEBUFFER, OpenGL.GL.GL_DEPTH_STENCIL_ATTACHMENT, OpenGL.GL.GL_RENDERBUFFER, rbo)
        assert (OpenGL.GL.glCheckFramebufferStatus(OpenGL.GL.GL_FRAMEBUFFER) == OpenGL.GL.GL_FRAMEBUFFER_COMPLETE)
        OpenGL.GL.glBindFramebuffer(OpenGL.GL.GL_FRAMEBUFFER, 0)

        # configure second post-processing framebuffer
        intermediate_fbo = OpenGL.GL.glGenFramebuffers(1)
        OpenGL.GL.glBindFramebuffer(OpenGL.GL.GL_FRAMEBUFFER, intermediate_fbo)
        # create a color attachment texture
        screen_texture = OpenGL.GL.glGenTextures(1)
        OpenGL.GL.glBindTexture(OpenGL.GL.GL_TEXTURE_2D, screen_texture)
        OpenGL.GL.glTexImage2D(
            OpenGL.GL.GL_TEXTURE_2D, 0, OpenGL.GL.GL_RGBA, width, height, 0,
            OpenGL.GL.GL_RGBA, OpenGL.GL.GL_UNSIGNED_BYTE, 0)
        OpenGL.GL.glTexParameteri(OpenGL.GL.GL_TEXTURE_2D, OpenGL.GL.GL_TEXTURE_MIN_FILTER, OpenGL.GL.GL_LINEAR)
        OpenGL.GL.glTexParameteri(OpenGL.GL.GL_TEXTURE_2D, OpenGL.GL.GL_TEXTURE_MAG_FILTER, OpenGL.GL.GL_LINEAR)
        OpenGL.GL.glFramebufferTexture2D(
            OpenGL.GL.GL_FRAMEBUFFER, OpenGL.GL.GL_COLOR_ATTACHMENT0, OpenGL.GL.GL_TEXTURE_2D,
            screen_texture, 0)  # we only need a color buffer
        assert (OpenGL.GL.glCheckFramebufferStatus(OpenGL.GL.GL_FRAMEBUFFER) == OpenGL.GL.GL_FRAMEBUFFER_COMPLETE)
        OpenGL.GL.glBindFramebuffer(OpenGL.GL.GL_FRAMEBUFFER, 0)

        OpenGL.GL.glBindFramebuffer(OpenGL.GL.GL_FRAMEBUFFER, framebuffer)

        # Clear the buffer
        OpenGL.GL.glClearColor(
            self.background_color[0],
            self.background_color[1],
            self.background_color[2],
            0.0)
        OpenGL.GL.glClear(OpenGL.GL.GL_COLOR_BUFFER_BIT | OpenGL.GL.GL_DEPTH_BUFFER_BIT)
        # Save old viewport
        viewport_ori = self.viewport
        self.viewport = numpy.array([0, 0, width, height])
        # Draw
        self.draw(data, update_matrices)
        # Restore viewport
        self.viewport = viewport_ori

        OpenGL.GL.glBindFramebuffer(OpenGL.GL.GL_READ_FRAMEBUFFER, framebuffer)
        OpenGL.GL.glBindFramebuffer(OpenGL.GL.GL_DRAW_FRAMEBUFFER, intermediate_fbo)
        OpenGL.GL.glBlitFramebuffer(
            0, 0, width, height, 0, 0, width, height, OpenGL.GL.GL_COLOR_BUFFER_BIT, OpenGL.GL.GL_NEAREST)

        OpenGL.GL.glBindFramebuffer(OpenGL.GL.GL_FRAMEBUFFER, intermediate_fbo)

        # Copy back in the given matrices
        pixels = OpenGL.GL.glReadPixels(0, 0, width, height, OpenGL.GL.GL_RGBA, OpenGL.GL.GL_UNSIGNED_BYTE)

        # Clean up
        OpenGL.GL.glBindFramebuffer(OpenGL.GL.GL_DRAW_FRAMEBUFFER, 0)
        OpenGL.GL.glBindFramebuffer(OpenGL.GL.GL_READ_FRAMEBUFFER, 0)
        OpenGL.GL.glBindFramebuffer(OpenGL.GL.GL_FRAMEBUFFER, 0)
        OpenGL.GL.glDeleteTextures(1, [screen_texture])
        OpenGL.GL.glDeleteTextures(1, [texture_color_buffer_multi_sampled])
        OpenGL.GL.glDeleteFramebuffers(1, [framebuffer])
        OpenGL.GL.glDeleteFramebuffers(1, [intermediate_fbo])
        OpenGL.GL.glDeleteRenderbuffers(1, [rbo])

        r = numpy.empty((width, height))
        g = numpy.empty((width, height))
        b = numpy.empty((width, height))
        a = numpy.empty((width, height))

        count = 0
        for j in range(height):
            for i in range(width):
                r[i, j] = pixels[count * 4 + 0]
                g[i, j] = pixels[count * 4 + 1]
                b[i, j] = pixels[count * 4 + 2]
                a[i, j] = pixels[count * 4 + 3]
                count += 1

        return r, g, b, a

    def draw_labels(self, data, labels):

        OpenGL.GL.glDisable(OpenGL.GL.GL_LINE_SMOOTH)  # Clear settings if overlay is activated
        data.meshgl.bind_labels(labels)
        viewi = OpenGL.GL.glGetUniformLocation(data.meshgl.shader_text, "view")
        proji = OpenGL.GL.glGetUniformLocation(data.meshgl.shader_text, "proj")
        OpenGL.GL.glUniformMatrix4fv(viewi, 1, OpenGL.GL.GL_FALSE, numpy.array(self.view, dtype=numpy.float32))
        OpenGL.GL.glUniformMatrix4fv(proji, 1, OpenGL.GL.GL_FALSE, numpy.array(self.proj, dtype=numpy.float32))
        # Parameters for mapping characters from font atlass
        width = self.viewport[2]
        height = self.viewport[3]
        if self.orthographic:
            text_shift_scale_factor = 0.01
            render_scale = 0.6
        else:
            text_shift_scale_factor = 0.03
            render_scale = 1.7
        OpenGL.GL.glUniform1f(
            OpenGL.GL.glGetUniformLocation(data.meshgl.shader_text, "TextShiftFactor"),
            text_shift_scale_factor)
        OpenGL.GL.glUniform3f(
            OpenGL.GL.glGetUniformLocation(data.meshgl.shader_text, "TextColor"),
            0.0, 0.0, 0.0)
        OpenGL.GL.glUniform2f(
            OpenGL.GL.glGetUniformLocation(data.meshgl.shader_text, "CellSize"),
            1.0 / 16.0, (300.0 / 384.0) / 6.0)
        OpenGL.GL.glUniform2f(
            OpenGL.GL.glGetUniformLocation(data.meshgl.shader_text, "CellOffset"),
            0.5 / 256.0, 0.5 / 256.0)
        OpenGL.GL.glUniform2f(
            OpenGL.GL.glGetUniformLocation(data.meshgl.shader_text, "RenderSize"),
            render_scale * 0.75 * 16.0 / float(width),
            render_scale * 0.75 * 33.33 / float(height))
        OpenGL.GL.glUniform2f(
            OpenGL.GL.glGetUniformLocation(data.meshgl.shader_text, "RenderOrigin"),
            -2.0, 2.0)
        data.meshgl.draw_labels(labels)
        OpenGL.GL.glEnable(OpenGL.GL.GL_DEPTH_TEST)

    def set_rotation_type(self, value):
        old_rotation_type = self.rotation_type
        self.rotation_type = value
        if (self.rotation_type == RotationType.ROTATION_TYPE_TWO_AXIS_VALUATOR_FIXED_UP and
                old_rotation_type != RotationType.ROTATION_TYPE_TWO_AXIS_VALUATOR_FIXED_UP):
            self.trackball_angle = snap_to_fixed_up(self.trackball_angle)

    def set(self, property_mask, value):
        if not value:
            return self.unset(property_mask)
        else:
            return property_mask | self.id

    def unset(self, property_mask):
        return property_mask & ~self.id

    def toggle(self, property_mask):
        return property_mask ^ self.id

    def is_set(self, property_mask):
        return property_mask & self.id
