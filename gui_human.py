import glfw
import glm
import OpenGL.GL as gl

import imgui
from imgui.integrations.glfw import GlfwRenderer

from lib.config import cfg

from lib.networks import make_network
from lib.datasets.make_dataset import make_dataset
from lib.utils.net_utils import load_network
from lib.utils.net_utils import perf_timer
from lib.utils.data_utils import to_cuda
from lib.visualizers import make_visualizer

import tqdm
import torch
import time
import ctypes
import numpy as np

from termcolor import colored
from lib.interactive import opt
from lib.interactive.camera import Camera

timer = perf_timer(use_ms=True, disabled=True)


class Renderer:
    def __init__(self):

        # load network
        print(colored('Loading network...', 'yellow'))
        self.network = make_network(cfg).cuda()
        load_network(self.network, cfg.trained_model_dir, epoch=cfg.test.epoch)
        self.network.eval()

        # prepare visualizer
        self.visualizer = make_visualizer(cfg)

        # prepare dataset
        print(colored('Loading dataset...', 'yellow'))
        self.dataset = make_dataset(cfg, is_train=False)  # proof of concept

        # see cas_enerf_render.py in ls dataset for example
        # self.frame_start = cfg.test_dataset.render_frames[0]
        # self.frame_step = cfg.test_dataset.render_frames[2]

        self.frame_start = cfg.test_dataset.frames[0]
        self.frame_step = cfg.test_dataset.frames[2]

        self.frame_cnt = len(self.dataset)
        self.iter = 0

        # prepare camera
        worldup, front, center = self.dataset.get_camera_up_front_center()
        center_tck, center_u, front_tck, front_u, worldup_tck, worldup_u = self.dataset.get_camera_tck(smoothing_term=opt.smoothing_term)
        self.cam = Camera(worldup=worldup,
                          front=front,
                          center=center,
                          center_tck=center_tck,
                          front_tck=front_tck,
                          worldup_tck=worldup_tck)

        # prepare texture for memory movement
        self.height, self.width = opt.window_hw
        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, self.width, self.height, 0,
                        gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

        self.readFboId = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.readFboId)
        gl.glFramebufferTexture2D(gl.GL_READ_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                                  gl.GL_TEXTURE_2D, self.tex, 0)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)

    @property
    def frame_curr(self):
        return self.frame_start + self.iter * self.frame_step

    def render(self):
        timer.logtime('Beginning frame...', logf=lambda x: print(colored(x, 'blue')))
        ret = self.render_next()  # TODO: now only a proof of concept
        img = ret[opt.type_mapping[opt.type]]
        img *= 255
        img = img.to(torch.uint8)
        img = torch.flip(img, (0,))
        img = img.cpu().numpy()  # from cpu
        ptr = img.data
        timer.logtime('GPU->MEM: {:.3f}')

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.width, self.height,
                           gl.GL_RGB, gl.GL_UNSIGNED_BYTE, ptr)  # to gpu, might slow down?
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.readFboId)  # write buffer defaults to 0
        gl.glBlitFramebuffer(0, 0, self.width, self.height,
                             0, 0, self.width, self.height,
                             gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)

        timer.logtime('MEM->GL: {:.3f}')

    def render_next(self):
        batch = self.dataset[(self.frame_curr, self.cam.get_c2w(), self.cam.get_w2c())]
        if opt.autoplay:
            self.iter += 1
            self.iter %= self.frame_cnt

        timer.logtime('MEM->GPU->RAYS: {:.3f}')
        with torch.no_grad():
            ret = self.network(batch)
        timer.logtime('NETWORK: {:.3f}')
        ret = self.visualizer.visualize(ret, batch)

        return ret


def draw_imgui(font, rend: Renderer):
    cam: Camera = rend.cam

    imgui.new_frame()

    with imgui.font(font):
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):

                clicked_quit, selected_quit = imgui.menu_item(
                    "Quit", 'Ctrl+Q', False, True
                )

                if clicked_quit:
                    exit(1)

                imgui.end_menu()
            imgui.end_main_menu_bar()

        imgui.begin(f"Render Backend: {next(rend.network.parameters()).device}")

        if imgui.collapsing_header('Camera', flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            # this returns, changed and values
            cam.center = glm.vec3(*imgui.input_float3('Center', *cam.center)[1])
            cam.origin = glm.vec3(*imgui.input_float3('Origin', *cam.origin)[1])
            cam.v_front = glm.vec3(*imgui.input_float3('Front', *cam.v_front)[1])
            if cam.has_cam_path:

                onpath_changed, cam.on_cam_path = imgui.checkbox('Snap To Path', cam.on_cam_path)

                if cam.on_cam_path:

                    # reload camera if checking snap to path
                    if onpath_changed:
                        cam.update_from_cam_path()

                    # update interpolation based on smoothing term setting
                    smoothing_changed, opt.smoothing_term = imgui.slider_float("Smoothing", opt.smoothing_term, 0, 1.0)

                    cam_path_u_changed, cam.cam_path_u = imgui.slider_float("Camera Rail", cam.cam_path_u, 0, 1.0)

                    # reload camera if smoothness changed
                    if smoothing_changed:
                        cam.center_tck, _, cam.front_tck, _, cam.worldup_tck, _ = rend.dataset.get_camera_tck(smoothing_term=opt.smoothing_term)
                        cam.update_from_cam_path()

                    if cam_path_u_changed:
                        cam.update_from_cam_path()

            # opt.lock_fxfy = imgui.checkbox('Lock fx=fy', opt.lock_fxfy)[1]

            # if opt.lock_fxfy:
            #     cam.fx = imgui.slider_float("Focal", cam.fx, 300, 7000)[1]
            #     cam.fy = cam.fx
            # else:
            #     cam.fx = imgui.slider_float("fx", cam.fx, 300, 7000)[1]
            #     cam.fy = imgui.slider_float("fy", cam.fy, 300, 7000)[1]

            if imgui.tree_node("Directions", flags=imgui.TREE_NODE_DEFAULT_OPEN):
                cam.v_world_up = glm.normalize(glm.vec3(*imgui.input_float3('World Up', *cam.v_world_up)[1]))
                cam.v_front = glm.normalize(glm.vec3(*imgui.input_float3('Front', *cam.v_front)[1]))
                imgui.tree_pop()

        if imgui.collapsing_header('Render', flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            auto_changed, opt.autoplay = imgui.checkbox('Auto Play', opt.autoplay)
            iter_changed, rend.iter = imgui.slider_int("Frame Index", rend.iter, 0, len(rend.dataset)-1)

            type_changed, opt.type = imgui.listbox('Render Type', opt.type, opt.type_mapping)
            # levels = list(map(str, range(cfg.cas_config.num)))
            # opt.render_level = int(imgui.listbox('Render Level', opt.render_level, levels)[1])

        imgui.end()

    # imgui.show_test_window()
    imgui.render()


from collections import deque
times = deque(maxlen=10)
def glfw_update_title(window):
    # ! SHOULD ALWAYS INIT glfw_update_title.prev, fcnt, prev_frame
    curr = glfw.get_time()
    elapsed = curr - glfw_update_title.prev  # logged when title update

    ftime = curr - glfw_update_title.prev_frame  # logged when frame updates
    glfw_update_title.prev_frame = curr

    # print(colored(f"Frametime: {ftime*1000:.3f}", "red"))

    if elapsed > opt.fps_cnter_int:
        fps = glfw_update_title.fcnt / elapsed
        times.append(fps)
        if len(times) == 10:
            title = f"Efficient NeRF: {np.mean(sorted(np.array(times))[2:-2]):.3f} fps"
        else:
            title = f"Efficient NeRF: {fps:.3f} fps"
        # print(title)
        glfw.set_window_title(window, f"{title}")
        glfw_update_title.fcnt = 0
        glfw_update_title.prev = curr

    glfw_update_title.fcnt += 1


def main():
    window = impl_glfw_init()  # prepare gl bindings
    impl = imgui_init(window)  # prepare imgui related init
    font = imgui_load_font(impl, opt.font_filepath, 14)  # prepare gui font
    rend = Renderer()  # prepare network and dataloader

    glfw_bind_callback(window, rend)  # keyboard, mouse and resizing

    while not glfw.window_should_close(window):
        rend.render()  # render network output to main frame buffer

        draw_imgui(font, rend)  # defines GUI elements
        impl.render(imgui.get_draw_data())  # render actual GUI elements
        impl.process_inputs()  # keyboard and mouse inputs for imgui update

        glfw_update_title(window)  # update fps counter if needed
        glfw.swap_buffers(window)
        glfw.poll_events()  # process pending events, keyboard and stuff
        timer.logtime('Other: {:.3f}')

    impl.shutdown()
    glfw.terminate()


def glfw_char_callback(window, codepoint):
    if (imgui.get_io().want_capture_keyboard):
        return
    char = chr(codepoint)
    rend: Renderer = glfw.get_window_user_pointer(window)

    if char <= '9' and char >= '0':
        index = int(char) - 1
        rend.cam.v_world_up, rend.cam.v_front, rend.cam.center = rend.dataset.get_camera_up_front_center(index=index)
        rend.cam.update_trans()


def glfw_key_callback(window, key, scancode, action, mods):
    if (imgui.get_io().want_capture_keyboard):
        return

    rend: Renderer = glfw.get_window_user_pointer(window)
    # s for snap
    if action == glfw.PRESS and key == glfw.KEY_S:
        rend.cam.v_world_up, rend.cam.v_front, rend.cam.center = rend.dataset.get_camera_up_front_center(rend.dataset.get_closest_camera(rend.cam.center))
        rend.cam.update_trans()

    # u for up
    if action == glfw.PRESS and key == glfw.KEY_D:
        rend.cam.v_world_up = glm.vec3(np.round(rend.cam.v_world_up))
        rend.cam.update_trans()

    if action == glfw.PRESS and key == glfw.KEY_A:
        rend.cam.v_world_up, rend.cam.v_front, rend.cam.center = rend.dataset.get_camera_up_front_center(rend.dataset.get_closest_camera(rend.cam.center))
        rend.cam.v_world_up = glm.vec3(np.round(rend.cam.v_world_up))
        rend.cam.update_trans()

    if action == glfw.PRESS and key == glfw.KEY_SPACE:
        opt.autoplay = not opt.autoplay


def glfw_mouse_button_callback(window, button, action, mods):
    if (imgui.get_io().want_capture_mouse):
        return
    rend: Renderer = glfw.get_window_user_pointer(window)
    x, y = glfw.get_cursor_pos(window)
    if action == glfw.PRESS:
        SHIFT = mods & glfw.MOD_SHIFT
        CONTROL = mods & glfw.MOD_CONTROL
        rend.cam.begin_drag(
            x, y, SHIFT or button == glfw.MOUSE_BUTTON_MIDDLE,
            (button == glfw.MOUSE_BUTTON_RIGHT) or (button == glfw.MOUSE_BUTTON_MIDDLE and SHIFT),
            CONTROL
        )
    elif action == glfw.RELEASE:
        rend.cam.end_drag()


def glfw_cursor_pos_callback(window, x, y):
    cam = glfw.get_window_user_pointer(window).cam
    cam.drag_update(x, y)


def glfw_scroll_callback(window, xoffset, yoffset):
    if (imgui.get_io().want_capture_mouse):
        return
    cam = glfw.get_window_user_pointer(window).cam
    speed_factor = 1e-1
    movement = -speed_factor if yoffset < 0 else speed_factor
    cam.move(cam.v_front * movement)


def glfw_window_size_callback(window, width, height):
    pass


def glfw_bind_callback(window, rend):
    glfw.set_char_callback(window, glfw_char_callback)
    glfw.set_window_user_pointer(window, rend)  # set the user, for retrival
    glfw.set_key_callback(window, glfw_key_callback)
    glfw.set_mouse_button_callback(window, glfw_mouse_button_callback)
    glfw.set_cursor_pos_callback(window, glfw_cursor_pos_callback)
    glfw.set_scroll_callback(window, glfw_scroll_callback)
    glfw.set_framebuffer_size_callback(window, glfw_window_size_callback)


def imgui_init(window):
    imgui.create_context()
    impl = GlfwRenderer(window)  # show window when network's already prepared
    return impl


def imgui_load_font(impl, filepath, fontsize):
    io = imgui.get_io()
    font = io.fonts.add_font_from_file_ttf(
        filepath, fontsize,
    )
    impl.refresh_font_texture()
    return font


def impl_glfw_init():
    height, width = opt.window_hw
    window_name = "ENeRF"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    # OS X supports only forward-compatible core profiles from 3.2
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(
        int(width), int(height), window_name, None, None
    )
    glfw.make_context_current(window)
    glfw_update_title.prev_frame = 0
    glfw_update_title.prev = 0
    glfw_update_title.fcnt = 0

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window


if __name__ == "__main__":
    main()
