import time
import copy
import glfw
import OpenGL.GL
import platform
import tkinter
from tkinter import filedialog
import numpy
import quaternion
from enum import Enum
from utils.camera import project, unproject
from utils.camera import trackball, two_axis_evaluator_fixed_up
from utils.camera import snap_to_canonical_view_quat
from viewer.ViewerCore import ViewerCore, RotationType
from viewer.ViewerData import ViewerData


class MouseButton(Enum):
    Left = 0
    Middle = 1
    Right = 2


class MouseMode(Enum):
    NoMode = 0
    Rotation = 1
    Zoom = 2
    Pan = 3
    Translation = 4


def glfw_mouse_press(window, button, action, modifier):
    if button == glfw.MOUSE_BUTTON_1:
        mb = MouseButton.Left
    elif button == glfw.MOUSE_BUTTON_2:
        mb = MouseButton.Right
    else:  # if button == glfw.MOUSE_BUTTON_3:
        mb = MouseButton.Middle

    if action == glfw.PRESS:
        Viewer.the_viewer.mouse_down(mb, modifier)
    else:
        Viewer.the_viewer.mouse_up(mb, modifier)


def glfw_error_callback(error, description):
    print("Error:", description)


def glfw_char_mods_callback(window, codepoint, modifier):
    Viewer.the_viewer.key_pressed(codepoint, modifier)


def glfw_key_callback(window, key, scancode, action, modifier):
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, glfw.TRUE)

    if action == glfw.PRESS:
        Viewer.the_viewer.key_down(key, modifier)
    elif action == glfw.RELEASE:
        Viewer.the_viewer.key_up(key, modifier)
    elif action == glfw.REPEAT:
        Viewer.the_viewer.key_repeat(key, modifier)


def glfw_window_size(window, width, height):
    w = int(width * Viewer.highdpi)
    h = int(height * Viewer.highdpi)
    Viewer.the_viewer.post_resize(w, h)


def glfw_mouse_move(window, x, y):
    Viewer.the_viewer.mouse_move(int(x * Viewer.highdpi), int(y * Viewer.highdpi))


def glfw_mouse_scroll(window, x, y):
    Viewer.scroll_x += x
    Viewer.scroll_y += y
    Viewer.the_viewer.mouse_scroll(float(y))


def glfw_drop_callback(window, count, filenames):
    pass


class Viewer(object):
    the_viewer = None
    highdpi = 1
    scroll_x = 0
    scroll_y = 0
    first_time_hack = True

    def __init__(self):

        self.path = ""

        self.window = None

        self.data_list = []
        self.selected_data_index = -1
        self.next_data_id = 1

        self.core_list = []
        self.core_list.append(ViewerCore())
        self.selected_core_index = 0
        self.core_list[0].id = 1
        self.next_core_id = 2

        self.plugins = []

        # Temporary variables initialization
        self.mouse_mode = MouseMode.NoMode
        self.down_rotation = quaternion.from_rotation_matrix(numpy.identity(3))
        self.current_mouse_x = 0
        self.current_mouse_y = 0
        self.down_mouse_x = 0
        self.down_mouse_y = 0
        self.down_mouse_z = 0
        self.down_translation = numpy.array([0.0, 0.0, 0.0])
        self.down = False
        self.hack_never_moved = True
        self.scroll_position = 0.0

        # C++ style callbacks
        self.callback_init = None
        self.callback_pre_draw = None
        self.callback_post_draw = None
        self.callback_post_resize = None
        self.callback_mouse_down = None
        self.callback_mouse_up = None
        self.callback_mouse_move = None
        self.callback_mouse_scroll = None
        self.callback_key_pressed = None
        self.callback_key_down = None
        self.callback_key_up = None
        self.callback_key_repeat = None

    def launch(self, resizable=True, fullscreen=False, maximize=False, name='viewer', width=0, height=0):

        if not self.launch_init(resizable, fullscreen, maximize, name, width, height):
            self.launch_shut()
            return False

        self.launch_rendering(True)
        self.launch_shut()
        return True

    def launch_init(self, resizable=True, fullscreen=False, maximize=False, name='viewer',
                    window_width=0, window_height=0):

        glfw.set_error_callback(glfw_error_callback)
        if not glfw.init():
            print("Error: Could not initialize OpenGL context")
            return False

        glfw.window_hint(glfw.SAMPLES, 8)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        if platform.system() == "Darwin":  # __APPLE__
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

        if fullscreen:
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            self.window = glfw.create_window(mode.width, mode.height, name, monitor, None)
            window_width = mode.width
            # window_height = mode.height

        else:
            # Set default windows width
            if window_width <= 0:
                if len(self.core_list) == 1 and self.core().viewport[2] > 0:
                    window_width = self.core().viewport[2]
                else:
                    window_width = 1280

            # Set default windows height
            if window_height <= 0:
                if len(self.core_list) == 1 and self.core().viewport[3] > 0:
                    window_height = self.core().viewport[3]
                else:
                    window_height = 800

            self.window = glfw.create_window(window_width, window_height, name, None, None)
            if maximize:
                glfw.maximize_window(self.window)

        if self.window is None:
            print("Error: Could not create GLFW window")
            glfw.terminate()
            return False

        glfw.make_context_current(self.window)

        # Load OpenGL and its extensions
        # if not gladLoadGLLoader(glfwGetProcAddress)):
        #    print("Failed to load OpenGL and its extensions\n");
        #    return False

        major = glfw.get_window_attrib(self.window, glfw.CONTEXT_VERSION_MAJOR)
        minor = glfw.get_window_attrib(self.window, glfw.CONTEXT_VERSION_MINOR)
        rev = glfw.get_window_attrib(self.window, glfw.CONTEXT_REVISION)
        print("OpenGL version received: %d.%d.%d" % (major, minor, rev))
        print("Supported OpenGL is %s" % str(OpenGL.GL.glGetString(OpenGL.GL.GL_VERSION), "utf-8"))
        print("Supported GLSL is %s" % str(OpenGL.GL.glGetString(OpenGL.GL.GL_SHADING_LANGUAGE_VERSION), "utf-8"))

        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)

        # Initialize one and only reference
        Viewer.the_viewer = self

        # Register callbacks
        glfw.set_key_callback(self.window, glfw_key_callback)
        glfw.set_cursor_pos_callback(self.window, glfw_mouse_move)
        glfw.set_window_size_callback(self.window, glfw_window_size)
        glfw.set_mouse_button_callback(self.window, glfw_mouse_press)
        glfw.set_scroll_callback(self.window, glfw_mouse_scroll)
        glfw.set_char_mods_callback(self.window, glfw_char_mods_callback)
        glfw.set_drop_callback(self.window, glfw_drop_callback)

        # Handle retina displays(windows and mac)
        # width, height = glfw.get_framebuffer_size(self.window)
        width_window, height_window = glfw.get_window_size(self.window)
        Viewer.highdpi = window_width / width_window
        glfw.set_window_size(self.window, width_window, height_window)

        # Initialize viewer
        self.init()
        for core in self.core_list:
            for data in self.data_list:
                if data.is_visible & core.id:
                    self.core(core.id).align_camera_center(data.V, data.F)

        return True

    def launch_rendering(self, loop=True):

        # glfwMakeContextCurrent(window);
        # Rendering loop
        first = True
        num_extra_frames = 5
        frame_counter = 0
        while not glfw.window_should_close(self.window):

            tic = time.time()
            self.draw(first)
            if first:
                first = False
            glfw.swap_buffers(self.window)

            if self.core().is_animating or frame_counter < num_extra_frames:
                frame_counter += 1
                glfw.poll_events()
                # In microseconds
                toc = time.time()
                duration = (toc - tic)
                min_duration = 1.0 / self.core().animation_max_fps
                if duration < min_duration:
                    time.sleep(min_duration - duration)

            else:
                glfw.wait_events()
                frame_counter = 0

            if not loop:
                return

            if Viewer.first_time_hack:
                if platform.system() == "Darwin":  # __APPLE__
                    glfw.hide_window(self.window)
                    glfw.show_window(self.window)
                Viewer.first_time_hack = False

    def launch_shut(self):

        for data in self.data_list:
            data.meshgl.free()

        self.core().shut()  # Doesn't do anything
        self.shutdown_plugins()
        glfw.destroy_window(self.window)
        glfw.terminate()

    def init(self):

        self.core().init()  # Doesn't do anything

        if self.callback_init is not None:
            if self.callback_init():
                return

        self.init_plugins()

    def init_plugins(self):

        # Init all plugins
        for plugin in self.plugins:
            plugin.init(self)

    def shutdown_plugins(self):

        for plugin in self.plugins:
            plugin.shutdown()

    ###################################################################################################################
    # Mesh IO
    ###################################################################################################################

    def load(self, filename, only_vertices=False, camera=True):

        loaded = False

        for plugin in self.plugins:
            loaded = plugin.load(filename, only_vertices)
            if loaded:
                break

        if not loaded:
            return False

        if camera:
            for core in self.core_list:
                core.align_camera_center(self.data().V, self.data().F)

        for plugin in self.plugins:
            if plugin.post_load():
                return True

        return False

    def save(self, filename, only_vertices):

        for plugin in self.plugins:
            if plugin.save(filename, only_vertices):
                return True

        return False

    def unload(self):

        for plugin in self.plugins:
            if plugin.unload():
                return True

        return False

    ###################################################################################################################
    # Callbacks
    ###################################################################################################################

    def key_pressed(self, key, modifiers):

        for plugin in self.plugins:
            if plugin.key_pressed(key, modifiers):
                return True

        if self.callback_key_pressed is not None:
            if self.callback_key_pressed(key, modifiers):
                return True

        return False

    def key_down(self, key, modifiers):

        for plugin in self.plugins:
            if plugin.key_down(key, modifiers):
                return True

        if self.callback_key_down is not None:
            if self.callback_key_down(key, modifiers):
                return True

        return False

    def key_up(self, key, modifiers):

        for plugin in self.plugins:
            if plugin.key_up(key, modifiers):
                return True

        if self.callback_key_up is not None:
            if self.callback_key_up(key, modifiers):
                return True

        return False

    def key_repeat(self, key, modifiers):

        for plugin in self.plugins:
            if plugin.key_repeat(key, modifiers):
                return True

        if self.callback_key_repeat is not None:
            if self.callback_key_repeat(key, modifiers):
                return True

        return False

    def select_hovered_core(self):

        width_window, height_window = glfw.get_framebuffer_size(self.window)
        for i in range(len(self.core_list)):
            viewport = self.core_list[i].viewport

            if ((self.current_mouse_x > viewport[0]) and
                    (self.current_mouse_x < viewport[0] + viewport[2]) and
                    ((height_window - self.current_mouse_y) > viewport[1]) and
                    ((height_window - self.current_mouse_y) < viewport[1] + viewport[3])):
                self.selected_core_index = i
                break

    def mouse_down(self, button, modifier):

        # Remember mouse location at down even if used by callback / plugin
        self.down_mouse_x = self.current_mouse_x
        self.down_mouse_y = self.current_mouse_y

        for plugin in self.plugins:
            if plugin.mouse_down(button, modifier):
                return True

        if self.callback_mouse_down is not None:
            if self.callback_mouse_down(button, modifier):
                return True

        self.down = True

        # Select the core containing the click location
        self.select_hovered_core()

        self.down_translation = self.core().camera_translation

        # Initialization code for the trackball
        if self.selected_data_index < 0:
            center = numpy.array([0, 0, 0])
        elif self.data().V.shape[0] == 0:
            center = numpy.array([0, 0, 0])
        else:
            center = numpy.sum(self.data().V, axis=0) / self.data().V.shape[0]

        coord = project(center, self.core().view, self.core().proj, self.core().viewport)
        self.down_mouse_z = coord[2]
        self.down_rotation = self.core().trackball_angle

        self.mouse_mode = MouseMode.Rotation

        if button == MouseButton.Left:
            if self.core().rotation_type == RotationType.ROTATION_TYPE_NO_ROTATION:
                self.mouse_mode = MouseMode.Translation
            else:
                self.mouse_mode = MouseMode.Rotation
        elif button == MouseButton.Right:
            self.mouse_mode = MouseMode.Translation
        else:
            self.mouse_mode = MouseMode.NoMode

        return True

    def mouse_up(self, button, modifier):

        self.down = False

        for plugin in self.plugins:
            if plugin.mouse_up(button, modifier):
                return True

        if self.callback_mouse_up is not None:
            if self.callback_mouse_up(button, modifier):
                return True

        self.mouse_mode = MouseMode.NoMode

        return True

    def mouse_move(self, mouse_x, mouse_y):

        if self.hack_never_moved:
            self.down_mouse_x = mouse_x
            self.down_mouse_y = mouse_y
            self.hack_never_moved = False

        self.current_mouse_x = mouse_x
        self.current_mouse_y = mouse_y

        for plugin in self.plugins:
            if plugin.mouse_move(mouse_x, mouse_y):
                return True

        if self.callback_mouse_move is not None:
            if self.callback_mouse_move(mouse_x, mouse_y):
                return True

        if self.down:

            # We need the window height to transform the mouse click coordinates
            # into viewport-mouse-click coordinates for trackball and two_axis_valuator_fixed_up
            width_window, height_window = glfw.get_framebuffer_size(self.window)
            if self.mouse_mode == MouseMode.Rotation:
                if self.core().rotation_type == RotationType.ROTATION_TYPE_NO_ROTATION:
                    pass
                elif self.core().rotation_type == RotationType.ROTATION_TYPE_TRACKBALL:
                    self.core().trackball_angle = trackball(
                        self.core().viewport[2],
                        self.core().viewport[3],
                        2.0,
                        self.down_rotation,
                        self.down_mouse_x - self.core().viewport[0],
                        self.down_mouse_y - (height_window - self.core().viewport[1] - self.core().viewport[3]),
                        mouse_x - self.core().viewport[0],
                        mouse_y - (height_window - self.core().viewport[1] - self.core().viewport[3]))
                elif self.core().rotation_type == RotationType.ROTATION_TYPE_TWO_AXIS_VALUATOR_FIXED_UP:
                    self.core().trackball_angle = two_axis_evaluator_fixed_up(
                        self.core().viewport[2],
                        self.core().viewport[3],
                        2.0,
                        self.down_rotation,
                        self.down_mouse_x - self.core().viewport[0],
                        self.down_mouse_y - (height_window - self.core().viewport[1] - self.core().viewport[3]),
                        mouse_x - self.core().viewport[0],
                        mouse_y - (height_window - self.core().viewport[1] - self.core().viewport[3]))
                else:
                    assert (False and "Unknown rotation type")

            elif self.mouse_mode == MouseMode.Translation:
                # translation
                pos1 = unproject(
                    numpy.array([mouse_x, self.core().viewport[3] - mouse_y, self.down_mouse_z]),
                    self.core().view, self.core().proj, self.core().viewport)
                pos0 = unproject(
                    numpy.array([self.down_mouse_x, self.core().viewport[3] - self.down_mouse_y, self.down_mouse_z]),
                    self.core().view, self.core().proj, self.core().viewport)

                diff = pos1 - pos0
                self.core().camera_translation = self.down_translation + diff

            elif self.mouse_mode == MouseMode.Zoom:
                delta = 0.001 * (mouse_x - self.down_mouse_x + mouse_y - self.down_mouse_y)
                self.core().camera_zoom *= 1 + delta
                self.down_mouse_x = mouse_x
                self.down_mouse_y = mouse_y

            else:
                pass

        return True

    def mouse_scroll(self, delta_y):

        # Direct the scrolling operation to the appropriate viewport
        # (unless the core selection is locked by an ongoing mouse interaction)
        if not self.down:
            self.select_hovered_core()
        self.scroll_position += delta_y

        for plugin in self.plugins:
            if plugin.mouse_scroll(delta_y):
                return True

        if self.callback_mouse_scroll is not None:
            if self.callback_mouse_scroll(delta_y):
                return True

        # Only zoom if there's actually a change
        if delta_y != 0:
            if delta_y > 0:
                multiplier = 1.0 + 0.05
            else:
                multiplier = 1.0 - 0.05
            min_zoom = 0.1
            if self.core().camera_zoom * multiplier > min_zoom:
                self.core().camera_zoom = self.core().camera_zoom * multiplier
            else:
                self.core().camera_zoom = min_zoom

            return True

    def draw(self, first):

        width, height = glfw.get_framebuffer_size(self.window)
        width_window, height_window = glfw.get_window_size(self.window)

        if width_window == 0 or width == 0:
            highdpi_tmp = Viewer.highdpi
        else:
            highdpi_tmp = width / width_window

        if abs(highdpi_tmp - Viewer.highdpi) > 1e-8:
            self.post_resize(width, height)
            Viewer.highdpi = highdpi_tmp

        for core in self.core_list:
            core.clear_framebuffers()

        for plugin in self.plugins:
            if plugin.pre_draw(first):
                break

        if self.callback_pre_draw is not None:
            if self.callback_pre_draw():
                pass

        self.refresh()

        for plugin in self.plugins:
            if plugin.post_draw(first):
                break

        if self.callback_post_draw is not None:
            if self.callback_post_draw():
                pass

    def refresh(self):

        for core in self.core_list:
            if core.is_visible:
                for data in self.data_list:
                    if data.is_visible & core.id:
                        core.draw(data)

    def resize(self, w, h):
        if self.window:
            glfw.set_window_size(self.window, (w / Viewer.highdpi), (h / Viewer.highdpi))
        self.post_resize(w, h)

    def post_resize(self, w, h):
        if len(self.core_list) == 1:
            self.core().viewport = [0, 0, w, h]
        else:
            # It is up to the user to define the behavior of the post_resize() function when there are multiple
            # viewports (through the `callback_post_resize` callback)
            pass

        for plugin in self.plugins:
            plugin.post_resize(w, h)

        if self.callback_post_resize is not None:
            self.callback_post_resize(w, h)

    def snap_to_canonical_quaternion(self):
        snapq = self.core().trackball_angle
        self.core().trackball_angle = snap_to_canonical_view_quat(snapq, 1.0)

    def open_dialog_load_mesh(self):

        tkinter.Tk().withdraw()
        filename = filedialog.askopenfilename(initialdir=self.path)

        if len(filename) == 0:
            return False

        return self.load(filename, only_vertices=False)

    def open_dialog_save_mesh(self):

        tkinter.Tk().withdraw()
        filename = filedialog.asksaveasfilename(initialdir=self.path)

        if len(filename) == 0:
            return False

        return self.save(filename, only_vertices=False)

    def open_dialog_load_point_cloud(self):

        tkinter.Tk().withdraw()
        filename = filedialog.askopenfilename(initialdir=self.path)

        if len(filename) == 0:
            return False

        return self.load(filename, only_vertices=True)

    def open_dialog_save_point_cloud(self):

        tkinter.Tk().withdraw()
        filename = filedialog.asksaveasfilename(initialdir=self.path)

        if len(filename) == 0:
            return False

        return self.save(filename, only_vertices=True)

    ###################################################################################################################
    # Multi-mesh methods
    ###################################################################################################################

    def data(self, data_id=-1):

        if data_id == -1:
            index = self.selected_data_index
        else:
            index = data_id
        assert (0 <= index < len(self.data_list) and "selected_data_index or mesh_id should be in bounds")
        return self.data_list[index]

    def append_data(self, visible=True):

        self.data_list.append(ViewerData())
        self.selected_data_index = len(self.data_list) - 1
        self.data_list[-1].id = self.next_data_id
        self.next_data_id += 1
        if visible:
            for core in self.core_list:
                self.data_list[-1].set_visible(True, core.id)
        else:
            self.data_list[-1].is_visible = 0
        return self.data_list[-1].id

    def erase_data(self, index):

        assert (0 <= index < len(self.data_list) and "index should be in bounds")
        self.data_list[index].meshgl.free()
        self.data_list.pop(index)
        if self.selected_data_index >= index:
            self.selected_data_index -= 1
        return True

    ###################################################################################################################
    # Multi-viewport methods
    ###################################################################################################################

    def core(self, core_id=0):

        assert (len(self.core_list) != 0 and "core_list should never be empty")
        if core_id == 0:
            core_index = self.selected_core_index
        else:
            core_index = self.core_index(core_id)
        assert (0 <= core_index < len(self.core_list) and "selected_core_index should be in bounds")
        return self.core_list[core_index]

    def erase_core(self, index):

        assert (0 <= index < len(self.core_list) and "index should be in bounds")
        if len(self.core_list) == 1:
            # Cannot remove last viewport
            return False
        self.core_list[index].shut()  # does nothing
        self.core_list.pop(index)
        if self.selected_core_index >= index and self.selected_core_index > 0:
            self.selected_core_index -= 1
        return True

    def core_index(self, core_id):

        for i in range(len(self.core_list)):
            if self.core_list[i].id == core_id:
                return i
        return 0

    def append_core(self, viewport, append_empty=False):
        # copies the previous active core and only changes the viewport
        self.core_list.append(copy.deepcopy(self.core()))
        self.core_list[-1].viewport = viewport
        self.core_list[-1].id = self.next_core_id
        self.next_core_id *= 2
        if not append_empty:
            for data in self.data_list:
                data.set_visible(True, self.core_list[-1].id)
                data.copy_options(self.core(), self.core_list[-1])

        self.selected_core_index = len(self.core_list) - 1
        return self.core_list[-1].id
