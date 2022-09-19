import numpy
import OpenGL.GL
from enum import IntFlag
from utils.VeraSansMono import VeraSansMono


class DirtyFlags(IntFlag):
    DIRTY_NONE = 0
    DIRTY_POSITION = 1
    DIRTY_UV = 2
    DIRTY_NORMAL = 4
    DIRTY_AMBIENT = 8
    DIRTY_DIFFUSE = 16
    DIRTY_SPECULAR = 32
    DIRTY_TEXTURE = 64
    DIRTY_FACE = 128
    DIRTY_MESH = 255
    DIRTY_OVERLAY_LINES = 256
    DIRTY_OVERLAY_POINTS = 512
    DIRTY_VERTEX_LABELS = 1024
    DIRTY_FACE_LABELS = 2048
    DIRTY_CUSTOM_LABELS = 4096
    DIRTY_ALL = 65535


class TextGL(object):

    def __init__(self):
        self.dirty_flag = DirtyFlags.DIRTY_ALL
        self.vao_labels = 0
        self.vbo_labels_pos = 0
        self.vbo_labels_characters = 0
        self.vbo_labels_offset = 0
        self.vbo_labels_indices = 0
        self.label_pos_vbo = numpy.empty(0, dtype=numpy.float32)
        self.label_char_vbo = numpy.empty(0, dtype=numpy.float32)
        self.label_offset_vbo = numpy.empty(0, dtype=numpy.float32)
        self.label_indices_vbo = numpy.empty(0, dtype=numpy.uint32)

    def init_buffers(self):
        self.vao_labels = OpenGL.GL.glGenVertexArrays(1)
        OpenGL.GL.glBindVertexArray(self.vao_labels)
        self.vbo_labels_pos = OpenGL.GL.glGenBuffers(1)
        self.vbo_labels_characters = OpenGL.GL.glGenBuffers(1)
        self.vbo_labels_offset = OpenGL.GL.glGenBuffers(1)
        self.vbo_labels_indices = OpenGL.GL.glGenBuffers(1)

    def free_buffers(self):
        OpenGL.GL.glDeleteBuffers(1, [self.vbo_labels_pos])
        OpenGL.GL.glDeleteBuffers(1, [self.vbo_labels_characters])
        OpenGL.GL.glDeleteBuffers(1, [self.vbo_labels_offset])
        OpenGL.GL.glDeleteBuffers(1, [self.vbo_labels_indices])


class MeshGL(object):

    def __init__(self):
        self.is_initialized = False

        self.vao_mesh = 0
        self.vao_overlay_lines = 0
        self.vao_overlay_points = 0
        self.shader_mesh = 0
        self.shader_overlay_lines = 0
        self.shader_overlay_points = 0
        self.shader_text = 0

        self.vbo_V = 0  # Vertices of the current mesh (#V x 3)
        self.vbo_V_uv = 0  # UV coordinates for the current mesh (#V x 2)
        self.vbo_V_normals = 0  # Vertices of the current mesh (#V x 3)
        self.vbo_V_ambient = 0  # Ambient material (#V x 3)
        self.vbo_V_diffuse = 0  # Diffuse material (#V x 3)
        self.vbo_V_specular = 0  # Specular material (#V x 3)

        self.vbo_F = 0  # Faces of the mesh (#F x 3)
        self.vbo_tex = 0  # Texture

        self.vbo_lines_F = 0  # Indices of the line overlay
        self.vbo_lines_V = 0  # Vertices of the line overlay
        self.vbo_lines_V_colors = 0  # Color values of the line overlay
        self.vbo_points_F = 0  # Indices of the point overlay
        self.vbo_points_V = 0  # Vertices of the point overlay
        self.vbo_points_V_colors = 0  # Color values of the point overlay

        # Temporary copy of the content of each VBO
        self.V_vbo = numpy.empty(0, dtype=numpy.float32)
        self.V_normals_vbo = numpy.empty(0, dtype=numpy.float32)
        self.V_ambient_vbo = numpy.empty(0, dtype=numpy.float32)
        self.V_diffuse_vbo = numpy.empty(0, dtype=numpy.float32)
        self.V_specular_vbo = numpy.empty(0, dtype=numpy.float32)
        self.V_uv_vbo = numpy.empty(0, dtype=numpy.float32)

        self.lines_V_vbo = numpy.empty(0, dtype=numpy.float32)
        self.lines_V_colors_vbo = numpy.empty(0, dtype=numpy.float32)
        self.points_V_vbo = numpy.empty(0, dtype=numpy.float32)
        self.points_V_colors_vbo = numpy.empty(0, dtype=numpy.float32)

        self.F_vbo = numpy.empty(0, dtype=numpy.uint32)
        self.lines_F_vbo = numpy.empty(0, dtype=numpy.uint32)
        self.points_F_vbo = numpy.empty(0, dtype=numpy.uint32)

        self.vertex_labels = TextGL()
        self.face_labels = TextGL()
        self.custom_labels = TextGL()
        self.font_atlas_id = 0

        self.tex_u = -1
        self.tex_v = -1
        self.tex_filter = OpenGL.GL.GL_LINEAR
        self.tex_wrap = OpenGL.GL.GL_REPEAT
        self.tex = numpy.empty(0, dtype=numpy.uint8)

        # Marks dirty buffers that need to be uploaded to OpenGL
        self.dirty = DirtyFlags.DIRTY_ALL

    def init_buffers(self):

        # Mesh: Vertex Array Object & Buffer objects
        self.vao_mesh = OpenGL.GL.glGenVertexArrays(1)
        OpenGL.GL.glBindVertexArray(self.vao_mesh)
        self.vbo_V = OpenGL.GL.glGenBuffers(1)
        self.vbo_V_normals = OpenGL.GL.glGenBuffers(1)
        self.vbo_V_ambient = OpenGL.GL.glGenBuffers(1)
        self.vbo_V_diffuse = OpenGL.GL.glGenBuffers(1)
        self.vbo_V_specular = OpenGL.GL.glGenBuffers(1)
        self.vbo_V_uv = OpenGL.GL.glGenBuffers(1)
        self.vbo_F = OpenGL.GL.glGenBuffers(1)
        self.vbo_tex = OpenGL.GL.glGenTextures(1)
        self.font_atlas_id = OpenGL.GL.glGenTextures(1)

        # Line overlay
        self.vao_overlay_lines = OpenGL.GL.glGenVertexArrays(1)
        OpenGL.GL.glBindVertexArray(self.vao_overlay_lines)
        self.vbo_lines_F = OpenGL.GL.glGenBuffers(1)
        self.vbo_lines_V = OpenGL.GL.glGenBuffers(1)
        self.vbo_lines_V_colors = OpenGL.GL.glGenBuffers(1)

        # Point overlay
        self.vao_overlay_points = OpenGL.GL.glGenVertexArrays(1)
        OpenGL.GL.glBindVertexArray(self.vao_overlay_points)
        self.vbo_points_F = OpenGL.GL.glGenBuffers(1)
        self.vbo_points_V = OpenGL.GL.glGenBuffers(1)
        self.vbo_points_V_colors = OpenGL.GL.glGenBuffers(1)

        # Text Labels
        self.vertex_labels.init_buffers()
        self.face_labels.init_buffers()
        self.custom_labels.init_buffers()

        self.dirty = DirtyFlags.DIRTY_ALL

    def free_buffers(self):
        if self.is_initialized:
            OpenGL.GL.glDeleteVertexArrays(1, [self.vao_mesh])
            OpenGL.GL.glDeleteVertexArrays(1, [self.vao_overlay_lines])
            OpenGL.GL.glDeleteVertexArrays(1, [self.vao_overlay_points])

            OpenGL.GL.glDeleteBuffers(1, [self.vbo_V])
            OpenGL.GL.glDeleteBuffers(1, [self.vbo_V_normals])
            OpenGL.GL.glDeleteBuffers(1, [self.vbo_V_ambient])
            OpenGL.GL.glDeleteBuffers(1, [self.vbo_V_diffuse])
            OpenGL.GL.glDeleteBuffers(1, [self.vbo_V_specular])
            OpenGL.GL.glDeleteBuffers(1, [self.vbo_V_uv])
            OpenGL.GL.glDeleteBuffers(1, [self.vbo_F])
            OpenGL.GL.glDeleteBuffers(1, [self.vbo_lines_F])
            OpenGL.GL.glDeleteBuffers(1, [self.vbo_lines_V])
            OpenGL.GL.glDeleteBuffers(1, [self.vbo_lines_V_colors])
            OpenGL.GL.glDeleteBuffers(1, [self.vbo_points_F])
            OpenGL.GL.glDeleteBuffers(1, [self.vbo_points_V])
            OpenGL.GL.glDeleteBuffers(1, [self.vbo_points_V_colors])

            # Text Labels
            self.vertex_labels.free_buffers()
            self.face_labels.free_buffers()
            self.custom_labels.free_buffers()

            OpenGL.GL.glDeleteTextures(1, [self.vbo_tex])
            OpenGL.GL.glDeleteTextures(1, [self.font_atlas_id])

    @staticmethod
    def bind_vertex_attrib_array(program_shader, name, buffer_id, array, refresh):
        location = OpenGL.GL.glGetAttribLocation(program_shader, name)
        if location < 0:
            return location
        if array.size == 0:
            OpenGL.GL.glDisableVertexAttribArray(location)
            return location
        OpenGL.GL.glBindBuffer(OpenGL.GL.GL_ARRAY_BUFFER, buffer_id)
        if refresh:
            OpenGL.GL.glBufferData(
                OpenGL.GL.GL_ARRAY_BUFFER, array, OpenGL.GL.GL_DYNAMIC_DRAW)

        OpenGL.GL.glVertexAttribPointer(location, array.shape[1], OpenGL.GL.GL_FLOAT, OpenGL.GL.GL_FALSE, 0, None)
        OpenGL.GL.glEnableVertexAttribArray(location)
        return location

    def bind_mesh(self):

        OpenGL.GL.glBindVertexArray(self.vao_mesh)
        OpenGL.GL.glUseProgram(self.shader_mesh)
        MeshGL.bind_vertex_attrib_array(
            self.shader_mesh, "position",
            self.vbo_V, self.V_vbo,
            self.dirty & DirtyFlags.DIRTY_POSITION)
        MeshGL.bind_vertex_attrib_array(
            self.shader_mesh, "normal",
            self.vbo_V_normals, self.V_normals_vbo,
            self.dirty & DirtyFlags.DIRTY_NORMAL)
        MeshGL.bind_vertex_attrib_array(
            self.shader_mesh, "Ka",
            self.vbo_V_ambient, self.V_ambient_vbo,
            self.dirty & DirtyFlags.DIRTY_AMBIENT)
        MeshGL.bind_vertex_attrib_array(
            self.shader_mesh, "Kd",
            self.vbo_V_diffuse, self.V_diffuse_vbo,
            self.dirty & DirtyFlags.DIRTY_DIFFUSE)
        MeshGL.bind_vertex_attrib_array(
            self.shader_mesh, "Ks",
            self.vbo_V_specular, self.V_specular_vbo,
            self.dirty & DirtyFlags.DIRTY_SPECULAR)
        MeshGL.bind_vertex_attrib_array(
            self.shader_mesh, "texcoord",
            self.vbo_V_uv, self.V_uv_vbo,
            self.dirty & DirtyFlags.DIRTY_UV)

        OpenGL.GL.glBindBuffer(OpenGL.GL.GL_ELEMENT_ARRAY_BUFFER, self.vbo_F)
        if self.dirty & DirtyFlags.DIRTY_FACE:
            OpenGL.GL.glBufferData(
                OpenGL.GL.GL_ELEMENT_ARRAY_BUFFER, self.F_vbo, OpenGL.GL.GL_DYNAMIC_DRAW)

        OpenGL.GL.glActiveTexture(OpenGL.GL.GL_TEXTURE0)
        OpenGL.GL.glBindTexture(OpenGL.GL.GL_TEXTURE_2D, self.vbo_tex)
        if self.dirty & DirtyFlags.DIRTY_TEXTURE:
            OpenGL.GL.glTexParameteri(OpenGL.GL.GL_TEXTURE_2D, OpenGL.GL.GL_TEXTURE_WRAP_S, self.tex_wrap)
            OpenGL.GL.glTexParameteri(OpenGL.GL.GL_TEXTURE_2D, OpenGL.GL.GL_TEXTURE_WRAP_T, self.tex_wrap)
            OpenGL.GL.glTexParameteri(OpenGL.GL.GL_TEXTURE_2D, OpenGL.GL.GL_TEXTURE_MIN_FILTER, self.tex_filter)
            OpenGL.GL.glTexParameteri(OpenGL.GL.GL_TEXTURE_2D, OpenGL.GL.GL_TEXTURE_MAG_FILTER, self.tex_filter)
            OpenGL.GL.glPixelStorei(OpenGL.GL.GL_UNPACK_ALIGNMENT, 1)
            OpenGL.GL.glTexImage2D(
                OpenGL.GL.GL_TEXTURE_2D, 0, OpenGL.GL.GL_RGBA, self.tex_u, self.tex_v, 0,
                OpenGL.GL.GL_RGBA, OpenGL.GL.GL_UNSIGNED_BYTE, self.tex)

        OpenGL.GL.glUniform1i(OpenGL.GL.glGetUniformLocation(self.shader_mesh, "tex"), 0)
        self.dirty &= ~DirtyFlags.DIRTY_MESH

    def bind_overlay_lines(self):

        is_dirty = self.dirty & DirtyFlags.DIRTY_OVERLAY_LINES

        OpenGL.GL.glBindVertexArray(self.vao_overlay_lines)
        OpenGL.GL.glUseProgram(self.shader_overlay_lines)

        MeshGL.bind_vertex_attrib_array(
            self.shader_overlay_lines, "position",
            self.vbo_lines_V, self.lines_V_vbo, is_dirty)
        MeshGL.bind_vertex_attrib_array(
            self.shader_overlay_lines, "color",
            self.vbo_lines_V_colors, self.lines_V_colors_vbo, is_dirty)

        OpenGL.GL.glBindBuffer(OpenGL.GL.GL_ELEMENT_ARRAY_BUFFER, self.vbo_lines_F)
        if is_dirty:
            OpenGL.GL.glBufferData(
                OpenGL.GL.GL_ELEMENT_ARRAY_BUFFER, self.lines_F_vbo, OpenGL.GL.GL_DYNAMIC_DRAW)

        self.dirty &= ~DirtyFlags.DIRTY_OVERLAY_LINES

    def bind_overlay_points(self):

        is_dirty = self.dirty & DirtyFlags.DIRTY_OVERLAY_POINTS

        OpenGL.GL.glBindVertexArray(self.vao_overlay_points)
        OpenGL.GL.glUseProgram(self.shader_overlay_points)

        MeshGL.bind_vertex_attrib_array(
            self.shader_overlay_points, "position",
            self.vbo_points_V, self.points_V_vbo, is_dirty)
        MeshGL.bind_vertex_attrib_array(
            self.shader_overlay_points, "color",
            self.vbo_points_V_colors, self.points_V_colors_vbo, is_dirty)

        OpenGL.GL.glBindBuffer(OpenGL.GL.GL_ELEMENT_ARRAY_BUFFER, self.vbo_points_F)
        if is_dirty:
            OpenGL.GL.glBufferData(
                OpenGL.GL.GL_ELEMENT_ARRAY_BUFFER, self.points_F_vbo, OpenGL.GL.GL_DYNAMIC_DRAW)

        self.dirty &= ~DirtyFlags.DIRTY_OVERLAY_POINTS

    def init_text_rendering(self):

        # Decompress the png of the font atlas
        font_atlas = VeraSansMono.decompress_atlas()

        # Bind atlas
        OpenGL.GL.glBindTexture(OpenGL.GL.GL_TEXTURE_2D, self.font_atlas_id)
        OpenGL.GL.glTexParameteri(OpenGL.GL.GL_TEXTURE_2D, OpenGL.GL.GL_TEXTURE_WRAP_S, OpenGL.GL.GL_CLAMP_TO_EDGE)
        OpenGL.GL.glTexParameteri(OpenGL.GL.GL_TEXTURE_2D, OpenGL.GL.GL_TEXTURE_WRAP_T, OpenGL.GL.GL_CLAMP_TO_EDGE)
        OpenGL.GL.glTexParameteri(OpenGL.GL.GL_TEXTURE_2D, OpenGL.GL.GL_TEXTURE_MIN_FILTER, OpenGL.GL.GL_NEAREST)
        OpenGL.GL.glTexParameteri(OpenGL.GL.GL_TEXTURE_2D, OpenGL.GL.GL_TEXTURE_MAG_FILTER, OpenGL.GL.GL_NEAREST)
        OpenGL.GL.glTexImage2D(OpenGL.GL.GL_TEXTURE_2D, 0, OpenGL.GL.GL_RED, 256, 256, 0, OpenGL.GL.GL_RED,
                               OpenGL.GL.GL_UNSIGNED_BYTE, font_atlas)

        # TextGL initialization
        self.vertex_labels.dirty_flag = DirtyFlags.DIRTY_VERTEX_LABELS
        self.face_labels.dirty_flag = DirtyFlags.DIRTY_FACE_LABELS
        self.custom_labels.dirty_flag = DirtyFlags.DIRTY_CUSTOM_LABELS

    def bind_labels(self, labels):

        is_dirty = self.dirty & labels.dirty_flag

        OpenGL.GL.glBindTexture(OpenGL.GL.GL_TEXTURE_2D, self.font_atlas_id)
        OpenGL.GL.glBindVertexArray(labels.vao_labels)
        OpenGL.GL.glUseProgram(self.shader_text)

        MeshGL.bind_vertex_attrib_array(
            self.shader_text, "position",
            labels.vbo_labels_pos, labels.label_pos_vbo, is_dirty)
        MeshGL.bind_vertex_attrib_array(
            self.shader_text, "character",
            labels.vbo_labels_characters,
            labels.label_char_vbo, is_dirty)
        MeshGL.bind_vertex_attrib_array(
            self.shader_text, "offset",
            labels.vbo_labels_offset, labels.label_offset_vbo, is_dirty)

        OpenGL.GL.glBindBuffer(OpenGL.GL.GL_ELEMENT_ARRAY_BUFFER, labels.vbo_labels_indices)

        if is_dirty:
            OpenGL.GL.glBufferData(
                OpenGL.GL.GL_ELEMENT_ARRAY_BUFFER, labels.label_indices_vbo, OpenGL.GL.GL_DYNAMIC_DRAW)

        self.dirty &= ~labels.dirty_flag

    def draw_mesh(self, solid):

        if solid:
            OpenGL.GL.glPolygonMode(OpenGL.GL.GL_FRONT_AND_BACK, OpenGL.GL.GL_FILL)
        else:
            OpenGL.GL.glPolygonMode(OpenGL.GL.GL_FRONT_AND_BACK, OpenGL.GL.GL_LINE)

        # Avoid Z-buffer fighting between filled triangles & wireframe lines
        if solid:
            OpenGL.GL.glEnable(OpenGL.GL.GL_POLYGON_OFFSET_FILL)
            OpenGL.GL.glPolygonOffset(1.0, 1.0)
        OpenGL.GL.glDrawElements(
            OpenGL.GL.GL_TRIANGLES, 3 * self.F_vbo.shape[0],
            OpenGL.GL.GL_UNSIGNED_INT, None)

        OpenGL.GL.glDisable(OpenGL.GL.GL_POLYGON_OFFSET_FILL)
        OpenGL.GL.glPolygonMode(OpenGL.GL.GL_FRONT_AND_BACK, OpenGL.GL.GL_FILL)

    def draw_overlay_lines(self):
        OpenGL.GL.glDrawElements(
            OpenGL.GL.GL_LINES, self.lines_F_vbo.shape[0],
            OpenGL.GL.GL_UNSIGNED_INT, None)

    def draw_overlay_points(self):
        OpenGL.GL.glDrawElements(
            OpenGL.GL.GL_POINTS, self.points_F_vbo.shape[0],
            OpenGL.GL.GL_UNSIGNED_INT, None)

    def draw_labels(self, labels):
        OpenGL.GL.glDrawElements(
            OpenGL.GL.GL_POINTS, labels.label_indices_vbo.shape[0],
            OpenGL.GL.GL_UNSIGNED_INT, None)

    def init(self):

        if self.is_initialized:
            return

        self.is_initialized = True

        mesh_vertex_shader_string = """
            #version 150
            uniform mat4 view;
            uniform mat4 proj;
            uniform mat4 normal_matrix;
            in vec3 position;
            in vec3 normal;
            out vec3 position_eye;
            out vec3 normal_eye;
            in vec4 Ka;
            in vec4 Kd;
            in vec4 Ks;
            in vec2 texcoord;
            out vec2 texcoordi;
            out vec4 Kai;
            out vec4 Kdi;
            out vec4 Ksi;
        
            void main()
            {
                position_eye = vec3 (view * vec4 (position, 1.0));
                normal_eye = vec3 (normal_matrix * vec4 (normal, 0.0));
                normal_eye = normalize(normal_eye);
                gl_Position = proj * vec4 (position_eye, 1.0); //proj * view * vec4(position, 1.0);
                Kai = Ka;
                Kdi = Kd;
                Ksi = Ks;
                texcoordi = texcoord;
            }
            """

        mesh_fragment_shader_string = """
            #version 150
            uniform mat4 view;
            uniform mat4 proj;
            uniform vec4 fixed_color;
            in vec3 position_eye;
            in vec3 normal_eye;
            uniform vec3 light_position_eye;
            vec3 Ls = vec3 (1, 1, 1);
            vec3 Ld = vec3 (1, 1, 1);
            vec3 La = vec3 (1, 1, 1);
            in vec4 Ksi;
            in vec4 Kdi;
            in vec4 Kai;
            in vec2 texcoordi;
            uniform sampler2D tex;
            uniform float specular_exponent;
            uniform float lighting_factor;
            uniform float texture_factor;
            uniform float matcap_factor;
            uniform float double_sided;
            out vec4 outColor;
            void main()
            {
                if (matcap_factor == 1.0f)
                {
                    vec2 uv = normalize(normal_eye).xy * 0.5 + 0.5;
                    outColor = texture(tex, uv);
                }
                else
                {
                    vec3 Ia = La * vec3(Kai);    // ambient intensity
                
                    vec3 vector_to_light_eye = light_position_eye - position_eye;
                    vec3 direction_to_light_eye = normalize (vector_to_light_eye);
                    float dot_prod = dot (direction_to_light_eye, normalize(normal_eye));
                    float clamped_dot_prod = abs(max (dot_prod, -double_sided));
                    vec3 Id = Ld * vec3(Kdi) * clamped_dot_prod;    // Diffuse intensity
                
                    vec3 reflection_eye = reflect (-direction_to_light_eye, normalize(normal_eye));
                    vec3 surface_to_viewer_eye = normalize (-position_eye);
                    float dot_prod_specular = dot (reflection_eye, surface_to_viewer_eye);
                    dot_prod_specular = float(abs(dot_prod)==dot_prod) * abs(max (dot_prod_specular, -double_sided));
                    float specular_factor = pow (dot_prod_specular, specular_exponent);
                    vec3 Is = Ls * vec3(Ksi) * specular_factor;    // specular intensity
                    vec4 color = vec4(lighting_factor * (Is + Id) + Ia + (1.0 - lighting_factor) * vec3(Kdi), 
                                      (Kai.a + Ksi.a + Kdi.a) / 3.0);
                    outColor = mix(vec4(1,1,1,1), texture(tex, texcoordi), texture_factor) * color;
                    if (fixed_color != vec4(0.0)) outColor = fixed_color;
                }
            }
            """

        overlay_vertex_shader_string = """ 
            #version 150
            uniform mat4 view;
            uniform mat4 proj;
            in vec3 position;
            in vec3 color;
            out vec3 color_frag;
            
            void main()
            {
                gl_Position = proj * view * vec4 (position, 1.0);
                color_frag = color;
            }
            """

        overlay_fragment_shader_string = """
            #version 150
            in vec3 color_frag;
            out vec4 outColor;
            void main()
            {
                outColor = vec4(color_frag, 1.0);
            }
            """

        overlay_point_fragment_shader_string = """
            #version 150
            in vec3 color_frag;
            out vec4 outColor;
            void main()
            {
                if (length(gl_PointCoord - vec2(0.5)) > 0.5)
                discard;
                outColor = vec4(color_frag, 1.0);
            }
            """

        text_vert_shader = """
            #version 330
            in vec3 position;
            in float character;
            in float offset;
            uniform mat4 view;
            uniform mat4 proj;
            out int vPosition;
            out int vCharacter;
            out float vOffset;
            void main()
            {
                vCharacter = int(character);
                vOffset = offset;
                vPosition = gl_VertexID;
                gl_Position = proj * view * vec4(position, 1.0);
            }
            """

        text_geom_shader = """
            #version 150 core
            layout(points) in;
            layout(triangle_strip, max_vertices = 4) out;
            out vec2 gTexCoord;
            uniform mat4 view;
            uniform mat4 proj;
            uniform vec2 CellSize;
            uniform vec2 CellOffset;
            uniform vec2 RenderSize;
            uniform vec2 RenderOrigin;
            uniform float TextShiftFactor;
            in int vPosition[1];
            in int vCharacter[1];
            in float vOffset[1];
            void main()
            {
                // Code taken from https://prideout.net/strings-inside-vertex-buffers
                // Determine the final quad's position and size:
                vec4 P = gl_in[0].gl_Position + vec4( vOffset[0]*TextShiftFactor, 0.0, 0.0, 0.0 ); // 0.04
                vec4 U = vec4(1, 0, 0, 0) * RenderSize.x; // 1.0
                vec4 V = vec4(0, 1, 0, 0) * RenderSize.y; // 1.0
                
                // Determine the texture coordinates:
                int letter = vCharacter[0]; // used to be the character
                letter = clamp(letter - 32, 0, 96);
                int row = letter / 16 + 1;
                int col = letter % 16;
                float S0 = CellOffset.x + CellSize.x * col;
                float T0 = CellOffset.y + 1 - CellSize.y * row;
                float S1 = S0 + CellSize.x - CellOffset.x;
                float T1 = T0 + CellSize.y;
                
                // Output the quad's vertices:
                gTexCoord = vec2(S0, T1); gl_Position = P - U - V; EmitVertex();
                gTexCoord = vec2(S1, T1); gl_Position = P + U - V; EmitVertex();
                gTexCoord = vec2(S0, T0); gl_Position = P - U + V; EmitVertex();
                gTexCoord = vec2(S1, T0); gl_Position = P + U + V; EmitVertex();
                EndPrimitive();
            }
            """

        text_frag_shader = """
            #version 330
            out vec4 outColor;
            in vec2 gTexCoord;
            uniform sampler2D font_atlas;
            uniform vec3 TextColor;
            void main()
            {
                float A = texture(font_atlas, gTexCoord).r;
                outColor = vec4(TextColor, A);
            }
            """

        self.init_buffers()
        self.init_text_rendering()
        self.shader_mesh = MeshGL.create_shader_program(
            geom_source="",
            vert_source=mesh_vertex_shader_string,
            frag_source=mesh_fragment_shader_string,
            attrib=None)
        self.shader_overlay_lines = MeshGL.create_shader_program(
            geom_source="",
            vert_source=overlay_vertex_shader_string,
            frag_source=overlay_fragment_shader_string,
            attrib=None)
        self.shader_overlay_points = MeshGL.create_shader_program(
            geom_source="",
            vert_source=overlay_vertex_shader_string,
            frag_source=overlay_point_fragment_shader_string,
            attrib=None)
        self.shader_text = MeshGL.create_shader_program(
            geom_source=text_geom_shader,
            vert_source=text_vert_shader,
            frag_source=text_frag_shader,
            attrib=None)

    def free(self):
        if self.is_initialized:
            if self.shader_mesh:
                MeshGL.destroy_shader_program(self.shader_mesh)
                self.shader_mesh = 0
            if self.shader_overlay_lines:
                MeshGL.destroy_shader_program(self.shader_overlay_lines)
                self.shader_overlay_lines = 0
            if self.shader_overlay_points:
                MeshGL.destroy_shader_program(self.shader_overlay_points)
                self.shader_overlay_points = 0
            if self.shader_text:
                MeshGL.destroy_shader_program(self.shader_text)
                self.shader_text = 0
            self.free_buffers()

    @staticmethod
    def create_shader_program(geom_source="", vert_source="", frag_source="", attrib=None):

        if len(vert_source) == 0 and len(frag_source) == 0:
            print("Error: create_shader_program() could not create shader program,"
                  " both .vert and .frag source given were empty")
            return False

        # create program
        shader_id = OpenGL.GL.glCreateProgram()
        if shader_id == 0:
            print("Error: create_shader_program() could not create shader program.")
            return False

        g = 0
        v = 0
        f = 0

        if len(geom_source) != 0:
            # load geometry shader
            g = MeshGL.load_shader(geom_source, OpenGL.GL.GL_GEOMETRY_SHADER)
            if g == 0:
                print("Error: geometry shader failed to compile.")
                return False
            OpenGL.GL.glAttachShader(shader_id, g)

        if len(vert_source) != 0:
            # load vertex shader
            v = MeshGL.load_shader(vert_source, OpenGL.GL.GL_VERTEX_SHADER)
            if v == 0:
                print("Error: vertex shader failed to compile.")
                return False
            OpenGL.GL.glAttachShader(shader_id, v)

        if len(frag_source) != 0:
            # load fragment shader
            f = MeshGL.load_shader(frag_source, OpenGL.GL.GL_FRAGMENT_SHADER)
            if f == 0:
                print("Error: fragment shader failed to compile.")
                return False
            OpenGL.GL.glAttachShader(shader_id, f)

        # loop over attributes
        if attrib is not None:
            for key, value in attrib.items():
                OpenGL.GL.glBindAttribLocation(shader_id, value, key)

        # Link program
        OpenGL.GL.glLinkProgram(shader_id)
        if g != 0:
            OpenGL.GL.glDetachShader(shader_id, g)
            OpenGL.GL.glDeleteShader(g)
        if v != 0:
            OpenGL.GL.glDetachShader(shader_id, v)
            OpenGL.GL.glDeleteShader(v)
        if f != 0:
            OpenGL.GL.glDetachShader(shader_id, f)
            OpenGL.GL.glDeleteShader(f)

        # print log if any
        MeshGL.print_program_info_log(shader_id)

        return shader_id

    @staticmethod
    def destroy_shader_program(program_id):

        # Don't try to destroy id == 0 (no shader program)
        if program_id == 0:
            print("Error: destroy_shader_program() id = %d"
                  " but must should be positive\n", program_id)
            return False

        shaders = OpenGL.GL.glGetAttachedShaders(program_id)
        error_code = MeshGL.report_gl_error("")
        if OpenGL.GL.GL_NO_ERROR != error_code:
            return False
        for shader_id in shaders:
            OpenGL.GL.glDetachShader(program_id, shader_id)
            OpenGL.GL.glDeleteShader(shader_id)

        # Now that all of the shaders are gone we can just delete the program
        OpenGL.GL.glDeleteProgram(program_id)
        return True

    @staticmethod
    def load_shader(shader_source, shader_type):

        if len(shader_source) == 0:
            return 0
        shader_id = OpenGL.GL.glCreateShader(shader_type)
        if shader_id == 0:
            print("Error: load_shader() failed to create shader.\n")
            return 0

        # Pass shader source string
        OpenGL.GL.glShaderSource(shader_id, [shader_source])
        OpenGL.GL.glCompileShader(shader_id)
        # Print info log (if any)
        MeshGL.print_shader_info_log(shader_id)
        return shader_id

    @staticmethod
    def glu_error_string(error_code):

        # http://stackoverflow.com/q/28485180/148668
        # gluErrorString was deprecated

        if OpenGL.GL.GL_NO_ERROR == error_code:
            return "no error"
        elif OpenGL.GL.GL_INVALID_ENUM == error_code:
            return "invalid enum"
        elif OpenGL.GL.GL_INVALID_VALUE == error_code:
            return "invalid value"
        elif OpenGL.GL.GL_INVALID_OPERATION == error_code:
            return "invalid operation"
        # ifndef GL_VERSION_3_0
        # elif OpenGL.GL.GL_STACK_OVERFLOW == error_code:
        #     return "stack overflow"
        # elif OpenGL.GL.GL_STACK_UNDERFLOW == error_code:
        #     return "stack underflow"
        # elif OpenGL.GL.GL_TABLE_TOO_LARGE == error_code:
        #     return "table too large"
        # endif
        elif OpenGL.GL.GL_OUT_OF_MEMORY == error_code:
            return "out of memory"
        # ifdef GL_EXT_framebuffer_object
        # elif OpenGL.GL.GL_INVALID_FRAMEBUFFER_OPERATION_EXT == error_code:
        #    return "invalid framebuffer operation"
        # endif
        else:
            return "unknown error code"

    @staticmethod
    def report_gl_error(message):
        error_code = OpenGL.GL.glGetError()
        if OpenGL.GL.GL_NO_ERROR != error_code:
            print("GL_ERROR: ")
            print("%s%s\n", message, MeshGL.glu_error_string(error_code))
        return error_code

    @staticmethod
    def print_shader_info_log(obj):
        info_log = OpenGL.GL.glGetShaderInfoLog(obj)
        if len(info_log):  # Only print if there is something in the log
            print(str(info_log, "utf-8"))

    @staticmethod
    def print_program_info_log(obj):
        info_log = OpenGL.GL.glGetProgramInfoLog(obj)
        if len(info_log):  # Only print if there is something in the log
            print(str(info_log, "utf-8"))
