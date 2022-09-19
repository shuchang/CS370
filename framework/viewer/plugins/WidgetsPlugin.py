import os
import glfw
import imgui
from OpenGL.error import GLError
from imgui.integrations.glfw import GlfwRenderer
from viewer.ViewerPlugin import ViewerPlugin
from viewer.imgui.imgui_fonts_droid_sans import droid_sans_compressed_data, droid_sans_compressed_size


class WidgetsPlugin(ViewerPlugin):
    context_ = None

    def __init__(self):
        super().__init__()
        self.impl = None
        self.widgets = []
        self.hidpi_scaling_ = -1.0
        self.pixel_ratio_ = -1.0

    def init(self, viewer):
        super().init(viewer)

        # Setup ImGui binding
        if self.viewer is not None:

            if WidgetsPlugin.context_ is None:
                # Single global context by default, but can be overridden by the user
                WidgetsPlugin.context_ = imgui.create_context()

            self.impl = GlfwRenderer(self.viewer.window, attach_callbacks=False)
            io = imgui.get_io()
            io.ini_file_name = b''
            imgui.style_colors_dark()
            style = imgui.get_style()
            style.frame_rounding = 5.0
            self.reload_font()

        for widget in self.widgets:
            widget.init(self.viewer)

    def shutdown(self):
        super(WidgetsPlugin, self).shutdown()
        for widget in self.widgets:
            widget.shutdown()

        # Cleanup
        self.impl.shutdown()

    def load(self, filename, only_vertices):
        return False

    def unload(self):
        return True

    def save(self, filename, only_vertices):
        return False

    def serialize(self, buffer):
        return False

    def deserialize(self, buffer):
        return False

    def post_load(self):
        for widget in self.widgets:
            if widget.post_load():
                # return True
                pass
        return False

    def pre_draw(self, first):

        glfw.poll_events()
        self.impl.process_inputs()

        # Check whether window dpi has changed
        scaling = self.new_hidpi_scaling()
        if abs(scaling - self.hidpi_scaling()) > 1e-5:
            self.reload_font()

        imgui.new_frame()

        for widget in self.widgets:
            if widget.pre_draw(first):
                return True

        return False

    def draw(self, first):
        x_scale, y_scale = glfw.get_window_content_scale(self.viewer.window)
        _x_size, _y_size = glfw.get_window_size(self.viewer.window)
        x_size = float(_x_size) / x_scale
        y_size = float(_y_size) / y_scale
        x_spacing = 10.0 / x_scale
        y_spacing = 10.0 / y_scale

        x = x_spacing
        y = y_spacing
        for widget in self.widgets:
            dim = widget.draw(first, self.menu_scaling(), x_size, y_size, x, y)
            # y += dim.w / y_scale + y_spacing
            x += dim.z / x_scale + x_spacing

    def post_draw(self, first):
        self.draw(first)

        imgui.render()
        try:
            self.impl.render(imgui.get_draw_data())
        except GLError:  # Dear ImGui raises error when unloading mesh
            pass

        for widget in self.widgets:
            if widget.post_draw(first):
                return True

        return False

    def post_resize(self, w, h):

        if self.context_ is not None:
            io = imgui.get_io()
            io.display_size = imgui.Vec2(float(w), float(h))

        for widget in self.widgets:
            if widget.post_resize(w, h):
                return True

        return False

    def mouse_down(self, button, modifier):
        for widget in self.widgets:
            if widget.mouse_down(button, modifier):
                return True

        # ImGui_ImplGlfw_MouseButtonCallback(mViewer->window, button, GLFW_PRESS, modifier);
        return imgui.get_io().want_capture_mouse

    def mouse_up(self, button, modifier):
        for widget in self.widgets:
            if widget.mouse_up(button, modifier):
                return False  # should not steal mouse up
        # return ImGui::GetIO().want_capture_mouse;
        # !! Should not steal mouse up
        return False

    def mouse_move(self, mouse_x, mouse_y):
        for widget in self.widgets:
            if widget.mouse_move(mouse_x, mouse_y):
                return True
        return imgui.get_io().want_capture_mouse

    def mouse_scroll(self, delta_y):
        for widget in self.widgets:
            if widget.mouse_scroll(delta_y):
                return True
        # ImGui_ImplGlfw_ScrollCallback(mViewer->window, 0.0, delta_y);
        return imgui.get_io().want_capture_mouse

    def key_pressed(self, key, modifiers):
        for widget in self.widgets:
            if widget.key_pressed(key, modifiers):
                return True
        # ImGui_ImplGlfw_CharCallback(nullptr, key);
        return imgui.get_io().want_capture_keyboard

    def key_down(self, key, modifiers):
        for widget in self.widgets:
            if widget.key_down(key, modifiers):
                return True
        # ImGui_ImplGlfw_KeyCallback(mViewer->window, key, 0, GLFW_PRESS, modifiers);
        return imgui.get_io().want_capture_keyboard

    def key_up(self, key, modifiers):
        for widget in self.widgets:
            if widget.key_up(key, modifiers):
                return True
        # ImGui_ImplGlfw_KeyCallback(mViewer->window, key, 0, GLFW_RELEASE, modifiers);
        return imgui.get_io().want_capture_keyboard

    def key_repeat(self, key, modifiers):
        for widget in self.widgets:
            if widget.key_repeat(key, modifiers):
                return True
        # ImGui_ImplGlfw_KeyCallback(mViewer->window, key, 0, GLFW_REPEAT, modifiers);
        return imgui.get_io().want_capture_keyboard

    def add(self, widget):
        self.widgets.append(widget)

    def reload_font(self, font_size=13):
        self.hidpi_scaling_ = self.new_hidpi_scaling()
        self.pixel_ratio_ = self.new_pixel_ratio()
        io = imgui.get_io()
        io.fonts.clear()
        path = os.path.join(os.getcwd(), 'framework', 'viewer', 'imgui', 'DroidSans.ttf')
        io.fonts.add_font_from_file_ttf(path, font_size * self.hidpi_scaling_)
        io.font_global_scale = 1.0 / self.pixel_ratio_
        self.impl.refresh_font_texture()

    def menu_scaling(self):
        return self.hidpi_scaling_ / self.pixel_ratio_

    def pixel_ratio(self):
        return self.pixel_ratio_

    def hidpi_scaling(self):
        return self.hidpi_scaling_

    @staticmethod
    def new_pixel_ratio():
        # Computes pixel ratio for hidpi devices
        window = glfw.get_current_context()
        x_buf_size, _ = glfw.get_framebuffer_size(window)
        x_win_size, _ = glfw.get_window_size(window)
        return float(x_buf_size) / float(x_win_size)

    @staticmethod
    def new_hidpi_scaling():
        # Computes scaling factor for hidpi devices
        window = glfw.get_current_context()
        x_scale, y_scale = glfw.get_window_content_scale(window)
        return 0.5 * (x_scale + y_scale)
