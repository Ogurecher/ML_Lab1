def manhattan_distance(x, y):
    return sum([abs(xi - yi) for xi, yi in zip(x, y)])


def euclidean_distance(x, y):
    return sum([(xi - yi) ** 2 for xi, yi in zip(x, y)]) ** (1 / 2)


def chebyshev_distance(x, y):
    return max([abs(xi - yi) for xi, yi in zip(x, y)])


def distance(x, y, distance_function_type):
    if distance_function_type == 'manhattan':
        return manhattan_distance(x, y)

    if distance_function_type == 'euclidean':
        return euclidean_distance(x, y)

    if distance_function_type == 'chebyshev':
        return chebyshev_distance(x, y)