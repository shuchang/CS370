import math
import copy
import numpy
import quaternion
from utils.maths import normalize


########################################################################################################################
# Rotation matrix helpers
########################################################################################################################

def rotation_matrix_from_vectors(vec1, vec2):
    # https://stackoverflow.com/a/59204638
    a, b = (vec1 / numpy.linalg.norm(vec1)).reshape(3), (vec2 / numpy.linalg.norm(vec2)).reshape(3)
    v = numpy.cross(a, b)
    c = numpy.dot(a, b)
    s = numpy.linalg.norm(v)
    if s == 0.0:
        return numpy.identity(3)
    kmat = numpy.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = numpy.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def rotation_matrix_from_axis_angle(axis, theta):
    # http://stackoverflow.com/users/190597/unutbu
    axis = numpy.asarray(axis)
    theta = numpy.asarray(theta)
    axis = axis / math.sqrt(numpy.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return numpy.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


########################################################################################################################
# Quaternion helpers
########################################################################################################################

SQRT_2_OVER_2 = 0.70710678118654757

NUM_CANONICAL_VIEW_QUAT = 24

CANONICAL_VIEW_QUAT = numpy.array([
    [0, 0, 0, 1],  # 0
    [0, 0, SQRT_2_OVER_2, SQRT_2_OVER_2],  # 1
    [0, 0, 1, 0],  # 2
    [0, 0, SQRT_2_OVER_2, -SQRT_2_OVER_2],  # 3

    [0, -1, 0, 0],  # 4
    [-SQRT_2_OVER_2, SQRT_2_OVER_2, 0, 0],  # 5
    [-1, 0, 0, 0],  # 6
    [-SQRT_2_OVER_2, -SQRT_2_OVER_2, 0, 0],  # 7

    [-0.5, -0.5, -0.5, 0.5],  # 8
    [0, -SQRT_2_OVER_2, 0, SQRT_2_OVER_2],  # 9
    [0.5, -0.5, 0.5, 0.5],  # 10
    [SQRT_2_OVER_2, 0, SQRT_2_OVER_2, 0],  # 11

    [SQRT_2_OVER_2, 0, -SQRT_2_OVER_2, 0],  # 12
    [0.5, 0.5, -0.5, 0.5],  # 13
    [0, SQRT_2_OVER_2, 0, SQRT_2_OVER_2],  # 14
    [-0.5, 0.5, 0.5, 0.5],  # 15

    [0, SQRT_2_OVER_2, SQRT_2_OVER_2, 0],  # 16
    [-0.5, 0.5, 0.5, -0.5],  # 17
    [-SQRT_2_OVER_2, 0, 0, -SQRT_2_OVER_2],  # 18
    [-0.5, -0.5, -0.5, -0.5],  # 19

    [-SQRT_2_OVER_2, 0, 0, SQRT_2_OVER_2],  # 20
    [-0.5, -0.5, 0.5, 0.5],  # 21
    [0, -SQRT_2_OVER_2, SQRT_2_OVER_2, 0],  # 22
    [0.5, -0.5, 0.5, -0.5]  # 23
])


def __quat_d(w, h):
    if abs(w) < abs(h):
        return abs(w) - 4
    else:
        return abs(h) - 4


def __quat_ix(x, w, h):
    return (2.0 * x - w - 1.0) / __quat_d(w, h)


def __quat_iy(y, w, h):
    return (-2.0 * y + h - 1.0) / __quat_d(w, h)


########################################################################################################################
# Camera functions
########################################################################################################################

def look_at(eye, center, up):
    f = normalize(center - eye)
    s = normalize(numpy.cross(f, up))
    u = numpy.cross(s, f)
    matrix = numpy.identity(4)
    matrix[0, 0] = s[0]
    matrix[0, 1] = s[1]
    matrix[0, 2] = s[2]
    matrix[1, 0] = u[0]
    matrix[1, 1] = u[1]
    matrix[1, 2] = u[2]
    matrix[2, 0] = -f[0]
    matrix[2, 1] = -f[1]
    matrix[2, 2] = -f[2]
    matrix[0, 3] = -numpy.dot(s, eye)
    matrix[1, 3] = -numpy.dot(u, eye)
    matrix[2, 3] = numpy.dot(f, eye)
    return matrix


def ortho(left, right, bottom, top, near_val, far_val):
    projection = numpy.identity(4)
    projection[0, 0] = 2.0 / (right - left)
    projection[1, 1] = 2.0 / (top - bottom)
    projection[2, 2] = -2.0 / (far_val - near_val)
    projection[0, 3] = -(right + left) / (right - left)
    projection[1, 3] = -(top + bottom) / (top - bottom)
    projection[2, 3] = -(far_val + near_val) / (far_val - near_val)
    return projection


def frustum(left, right, bottom, top, near_val, far_val):
    projection = numpy.zeros((4, 4))
    projection[0, 0] = (2.0 * near_val) / (right - left)
    projection[1, 1] = (2.0 * near_val) / (top - bottom)
    projection[0, 2] = (right + left) / (right - left)
    projection[1, 2] = (top + bottom) / (top - bottom)
    projection[2, 2] = -(far_val + near_val) / (far_val - near_val)
    projection[3, 2] = -1.0
    projection[2, 3] = -(2.0 * far_val * near_val) / (far_val - near_val)
    return projection


def snap_to_fixed_up(q):
    up = quaternion.as_rotation_matrix(q) @ numpy.array([0, 1, 0])
    proj_up = numpy.array([0, up[1], up[2]])
    if numpy.linalg.norm(proj_up) == 0:
        proj_up = numpy.array([0, 1, 0])
    proj_up = normalize(proj_up)
    matrix = rotation_matrix_from_vectors(up, proj_up)
    dq = quaternion.from_rotation_matrix(matrix)
    s = dq * q
    return s


def snap_to_canonical_view_quat(q, threshold):
    s = copy.deepcopy(q)
    # Normalize input quaternion
    if q.norm() == 0:
        return s
    qn = q.normalized()

    # 0.290019
    max_distance = 0.4
    min_distance = 2 * max_distance
    min_index = -1
    min_sign = 0
    # loop over canonical view quaternions
    for sign in range(-1, 2, 1):
        for i in range(NUM_CANONICAL_VIEW_QUAT):
            distance = 0.0
            # loop over coordinates
            qn_array = quaternion.as_float_array(qn)
            for j in range(4):
                distance += (qn_array[j] - sign * CANONICAL_VIEW_QUAT[i, j]) * (qn_array[j] - sign * CANONICAL_VIEW_QUAT[i, j])
                if min_distance > distance:
                    min_distance = distance
                    min_index = i
                    min_sign = sign

    if max_distance < min_distance:
        print("Error: found new max MIN_DISTANCE: %g\n"
              "PLEASE update snap_to_canonical_quat()",
              min_distance)

    assert (min_distance < max_distance)
    assert (min_index >= 0)

    if min_distance / max_distance <= threshold:

        # loop over coordinates
        s_array = [0] * 4
        for j in range(4):
            s_array[j] = min_sign * CANONICAL_VIEW_QUAT[min_index, j]
        s = quaternion.from_float_array(s_array)
        return s

    return s


def project(objs, model, proj, viewport):
    if objs.ndim == 1:
        tmp = numpy.array([objs[0], objs[1], objs[2], 1.0])
        tmp = model @ tmp
        tmp = proj @ tmp
        tmp = tmp / tmp[3]
        tmp = tmp * 0.5 + 0.5
        tmp[0] = tmp[0] * viewport[2] + viewport[0]
        tmp[1] = tmp[1] * viewport[3] + viewport[1]
        return tmp[0:3]

    elif objs.ndim == 2:
        assert (objs.shape[1] == 3)
        tmp = numpy.ones((objs.shape[0], 4))
        tmp[:, 0:3] = objs
        tmp = tmp * model.transpose() * proj.template.transpose()
        tmp = tmp / tmp[:, 3][:, numpy.newaxis]
        tmp = tmp * 0.5 + 0.5
        tmp[:, 0] = tmp[:, 0] * viewport[2] + viewport[0]
        tmp[:, 1] = tmp[:, 1] * viewport[3] + viewport[1]
        return tmp[:, 0:3]

    else:
        return None


def unproject(wins, model, proj, viewport):
    if wins.ndim == 1:
        inverse = numpy.linalg.inv(proj @ model)
        tmp = numpy.array([wins[0], wins[1], wins[2], 1.0])
        tmp[0] = (tmp[0] - viewport[0]) / viewport[2]
        tmp[1] = (tmp[1] - viewport[1]) / viewport[3]
        tmp = tmp * 2.0 - 1.0
        tmp = inverse @ tmp
        tmp = tmp / tmp[3]
        return tmp[0:3]

    elif wins.ndim == 2:
        assert (wins.shape[1] == 3)
        n = wins.shape[0]
        objs = numpy.zeros((n, 3))
        for i in range(n):
            inverse = numpy.linalg.inv(proj @ model)
            tmp = numpy.array([wins[i, 0], wins[i, 1], wins[i, 2], 1.0])
            tmp[0] = (tmp[0] - viewport[0]) / viewport[2]
            tmp[1] = (tmp[1] - viewport[1]) / viewport[3]
            tmp = tmp * 2.0 - 1.0
            tmp = inverse @ tmp
            tmp = tmp / tmp[3]
            objs[i, :] = tmp[0:3]
        return objs

    else:
        return None


def two_axis_evaluator_fixed_up(w, h, speed, down_quaternion, down_x, down_y, mouse_x, mouse_y):
    axis = numpy.array([0, 1, 0])
    angle = math.pi * float(mouse_x - down_x) / w * speed / 2.0
    q = down_quaternion * quaternion.from_rotation_matrix(
        rotation_matrix_from_axis_angle(normalize(axis), angle))
    q = q.normalized()

    axis = numpy.array([1, 0, 0])
    angle = math.pi * float(mouse_y - down_y) / h * speed / 2.0
    q = quaternion.from_rotation_matrix(
        rotation_matrix_from_axis_angle(normalize(axis), angle)) * q
    q = q.normalized()

    return q


def trackball(w, h, speed_factor, down_quaternion, down_x, down_y, mouse_x, mouse_y):
    assert (speed_factor > 0)

    original_x = __quat_ix(speed_factor * (down_x - w / 2) + w / 2, w, h)
    original_y = __quat_iy(speed_factor * (down_y - h / 2) + h / 2, w, h)

    x = __quat_ix(speed_factor * (mouse_x - w / 2) + w / 2, w, h)
    y = __quat_iy(speed_factor * (mouse_y - h / 2) + h / 2, w, h)

    z = 1
    n0 = math.sqrt(original_x * original_x + original_y * original_y + z * z)
    n1 = math.sqrt(x * x + y * y + z * z)

    if n0 > 0.0 and n1 > 0.0:

        v0 = numpy.array([original_x / n0, original_y / n0, z / n0])
        v1 = numpy.array([x / n1, y / n1, z / n1])
        axis = numpy.cross(v0, v1)
        sa = math.sqrt(numpy.dot(axis, axis))
        ca = numpy.dot(v0, v1)
        angle = math.atan2(sa, ca)
        if x * x + y * y > 1.0:
            angle *= 1.0 + 0.2 * (math.sqrt(x * x + y * y) - 1.0)
        q_rot = quaternion.from_rotation_matrix(rotation_matrix_from_axis_angle(axis, angle))

    else:
        return None

    n_q_orig = math.sqrt(down_quaternion.x * down_quaternion.x +
                         down_quaternion.y * down_quaternion.y +
                         down_quaternion.z * down_quaternion.z +
                         down_quaternion.w * down_quaternion.w)

    if abs(n_q_orig) > 0.0:
        q_orig = numpy.quaternion(
            down_quaternion.w / n_q_orig,
            down_quaternion.x / n_q_orig,
            down_quaternion.y / n_q_orig,
            down_quaternion.z / n_q_orig)
        q_res = q_rot * q_orig
        return q_res
    else:
        return q_rot
