import glm
import numpy as np
from termcolor import colored
from scipy import interpolate
PI = 3.1415926536


def logf(x, disabled=True):
    if disabled:
        return
    return print(colored(x, "green"))


def log_yellow(x, disabled=True):
    if disabled:
        return
    return print(colored(x, "yellow"))


def flatten_inner_dim(x):
    # only a incomplete utility function
    # doesn't check that outer dim is there
    if isinstance(x, list):
        if len(x) > 1:
            return [flatten_inner_dim(v) for v in x]
        else:
            return flatten_inner_dim(x[0])
    elif isinstance(x, np.ndarray):
        x = x.squeeze()  # no recursion needed for np array
        if x.size > 1:
            return np.array([flatten_inner_dim(v) for v in x])
        else:
            return x.item()


class Camera:
    def __init__(self, worldup=glm.vec3(0, 0, 1), front=glm.vec3(-0.10432957, -0.93850941, -0.32911311), center=glm.vec3(0.29572367, 3.29936877, 1.9140842), front_tck=None, center_tck=None, worldup_tck=None):
        # constants
        self.CLIP_NEAR = 1e-3

        # draggin state
        self.is_dragging = False
        self.is_panning = False
        self.about_origin = False
        self.fix_y = False
        self.drag_start = glm.vec2()
        self.drag_start_right = glm.vec3()
        self.drag_start_front = glm.vec3()
        self.drag_start_down = glm.vec3()
        self.drag_start_center = glm.vec3()
        self.drag_start_origin = glm.vec3()
        self.movement_speed = 1  # GUI move speed

        # interal states
        self.width = 512
        self.height = 512
        self.fx = 1111.1
        self.fy = 1111.1
        self.center = glm.vec3(center)
        self.v_front = glm.vec3(front)
        self.v_world_up = glm.vec3(worldup)
        self.origin = glm.vec3(0, 0, 0)
        self.K = glm.mat4()
        self.c2w = glm.mat4()
        self.w2c = glm.mat4()
        self.update_trans()
        self.update_K()

        # camera path control
        self.front_tck = front_tck
        self.center_tck = center_tck
        self.worldup_tck = worldup_tck
        # this option should control whether current rotation (handled by right click) is controlled by the predefined camera path
        # loaded from the dataset, and interpolated with B-spline interpolation
        self.on_cam_path = False
        self.cam_path_u = 0.  # the parameter [0, 1] controlling camera path (from first camera interpolated to last one)

    @property
    def has_cam_path(self):
        return self.front_tck is not None and self.center_tck is not None and self.worldup_tck is not None

    def update_trans(self):
        self.v_front = glm.normalize(self.v_front)
        self.v_right = glm.normalize(glm.cross(self.v_front, self.v_world_up))
        self.v_down = glm.cross(self.v_front, self.v_right)
        self.c2w[0].xyz = self.v_right
        self.c2w[1].xyz = self.v_down
        self.c2w[2].xyz = self.v_front
        self.c2w[3].xyz = self.center
        self.w2c = glm.affineInverse(self.c2w)
        logf(f"new c2w:\n{self.get_c2w()}")

    def update_K(self):
        # note that OpenGL is column major
        # and this is a ndc space K
        self.K = glm.mat4(self.fx / (0.5 * self.width), 0, 0, 0,
                          0, self.fy / (0.5 * self.height), 0, 0,
                          0, 0, -1, -1,
                          0, 0, -2 * self.CLIP_NEAR, 0)

    def begin_drag(self, x, y, is_pan, about_origin, fix_y):
        self.is_dragging = True
        self.drag_start = glm.vec2(x, y)
        self.drag_start_front = self.v_front
        self.drag_start_right = self.v_right
        self.drag_start_down = self.v_down
        self.drag_start_center = self.center
        self.drag_start_origin = self.origin
        self.is_panning = is_pan
        self.about_origin = about_origin
        self.fix_y = fix_y
        self.drag_cam_path_u = self.cam_path_u

        logf(f"begin dragging: {x}, {y}, is_pan: {is_pan}, about_origin: {about_origin}")

    def end_drag(self):
        self.is_dragging = False

    def update_from_cam_path(self):
        self.center = glm.vec3(*flatten_inner_dim(interpolate.splev(self.cam_path_u, self.center_tck)))
        self.v_front = glm.normalize(glm.vec3(*flatten_inner_dim(interpolate.splev(self.cam_path_u, self.front_tck))))
        self.v_world_up = glm.normalize(glm.vec3(*flatten_inner_dim(interpolate.splev(self.cam_path_u, self.worldup_tck))))
        self.update_trans()

    def drag_update(self, x, y):
        # preprocessing
        if not self.is_dragging:
            return
        drag_curr = glm.vec2(x, y)
        delta = drag_curr - self.drag_start
        delta *= self.movement_speed / max(self.height, self.width)
        if self.fix_y:
            delta.y = 0

        # actual draggin and update
        if self.has_cam_path and self.on_cam_path:
            diff = -delta.x
            # diff /= 50
            self.cam_path_u = self.drag_cam_path_u + diff  # TODO: SIGN OF THIS?
            self.cam_path_u %= 1.0
            log_yellow(f"updating cam_path_u: {self.cam_path_u}")
            self.update_from_cam_path()

        elif self.is_panning:
            diff = delta.x * self.drag_start_right + delta.y * self.drag_start_down
            self.center = self.drag_start_center + diff
            if self.about_origin:
                self.origin = self.drag_start_origin + diff

            logf(f"panning center: {self.center}")

        else:
            if self.about_origin:
                delta *= -1

            m = glm.mat4(1)
            m = glm.rotate(m, delta.x % 2 * PI, self.v_world_up)
            m = glm.rotate(m, delta.y, self.drag_start_right)
            self.v_front = m * self.drag_start_front

            if self.about_origin:
                self.center = glm.vec3(-m*glm.vec4(self.origin - self.drag_start_center, 1)) + self.origin

            logf(f"rotating v_front: {self.v_front}")

        # postprocessing
        self.update_trans()

    def move(self, xyz):
        # move center (also OK if you're draggin)
        delta = xyz * self.movement_speed
        self.center += delta
        if self.is_dragging:
            self.drag_start_center += delta
        self.update_trans()

        logf(f"moving center: {self.center}")

    def get_c2w(self):
        # gives in numpy array (row major)
        return np.array(self.c2w)

    def get_w2c(self):
        # gives in numpy array (row major)
        return np.array(self.w2c)
