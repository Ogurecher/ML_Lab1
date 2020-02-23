import math

def uniform_kernel(r):
    return 1 / 2 - (abs(r) // 1) if r < 1 else 0


def triangular_kernel(r):
    return (1 - abs(r)) if r < 1 else 0


def epanechnikov_kernel(r):
    return (3 / 4 * (1 - r ** 2)) if r < 1 else 0


def quartic_kernel(r):
    return (15 / 16 * (1 - r ** 2) ** 2) if r < 1 else 0


def triweight_kernel(r):
    return (35 / 32 * (1 - r ** 2) ** 3) if r < 1 else 0


def tricube_kernel(r):
    return (70 / 81 * (1 - abs(r ** 3)) ** 3) if r < 1 else 0


def gaussian_kernel(r):
    return 1 / (2 * math.pi) ** (1 / 2) * math.e ** (-(1 / 2) * r ** 2)


def cosine_kernel(r):
    return (math.pi / 4 * math.cos(math.pi / 2 * r)) if r < 1 else 0


def logistic_kernel(r):
    return 1 / (math.e ** r + 2 + math.e ** (-r))


def sigmoid_kernel(r):
    return 2 / math.pi * 1 / (math.e ** r + math.e ** (-r))


def kernel(r, kernel_function_type):
    if kernel_function_type == 'uniform':
        return uniform_kernel(r)

    if kernel_function_type == 'triangular':
        return triangular_kernel(r)

    if kernel_function_type == 'epanechnikov':
        return epanechnikov_kernel(r)

    if kernel_function_type == 'quartic':
        return quartic_kernel(r)

    if kernel_function_type == 'triweight':
        return triweight_kernel(r)

    if kernel_function_type == 'tricube':
        return tricube_kernel(r)

    if kernel_function_type == 'gaussian':
        return gaussian_kernel(r)

    if kernel_function_type == 'cosine':
        return cosine_kernel(r)

    if kernel_function_type == 'logistic':
        return logistic_kernel(r)

    if kernel_function_type == 'sigmoid':
        return sigmoid_kernel(r)