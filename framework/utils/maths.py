import math
import numpy


def static_variables(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def normalize(v):
    norm = numpy.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def clamp(t, _min=0.0, _max=1.0):
    return max(min(t, 1.0), 0.0)


@static_variables(v1=0, v2=0, s=0, phase=0)
def gaussian(mean, std):
    if gaussian.phase == 0:
        while gaussian.s >= 1.0 or gaussian.s == 0.0:
            gaussian.v1 = numpy.random.uniform(low=-1.0, high=1.0)
            gaussian.v2 = numpy.random.uniform(low=-1.0, high=1.0)
            gaussian.s = gaussian.v1 * gaussian.v1 + gaussian.v2 * gaussian.v2
        x = gaussian.v1 * math.sqrt(-2.0 * math.log(gaussian.s) / gaussian.s)

    else:
        x = gaussian.v2 * math.sqrt(-2.0 * math.log(gaussian.s) / gaussian.s)

    gaussian.phase = 1 - gaussian.phase

    return x * std + mean


def random_vector():
    x = 0
    y = 0
    length = 0.0
    while length > 1.0 or length == 0.0:
        x = numpy.random.uniform(low=-1.0, high=1.0)
        y = numpy.random.uniform(low=-1.0, high=1.0)
        length = x * x + y * y

    r = 2 * math.sqrt(1.0 - length)
    x *= r
    y *= r
    z = 1.0 - 2.0 * length

    return numpy.array([x, y, z])
