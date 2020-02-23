distance_functions = ['manhattan', 'euclidean', 'chebyshev']
kernel_functions = ['uniform', 'triangular', 'epanechnikov', 'quartic', 'triweight', 'tricube', 'gaussian', 'cosine',
                    'logistic', 'sigmoid']

window_types = [#'fixed',
                'variable']
window_widths = [0.01, 0.05, 0.1, 0.2]
window_neighbours = [1, 2, 3, 4, 5, 10, 20, 30, 40,  50, 100, 150,  200, 212]

best_combination = {'distance_function': 'manhattan', 'kernel_function': 'triweight', 'window_type': 'variable', 'window_parameter': 1}