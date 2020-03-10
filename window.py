import numpy as np

def get_window_at(contents, x, y, window_size=7):
    """Given the map in halite format, returns a window of `window_size` centered at `x` and `y`.
    The window is a np.ndarray of shape (window_size, window_size, 6), where the last dimension contains the
    different input channels:
        0: strength of own units
        1: strength of enemy units
        2: strength of neutral units
        3: production of own squares
        4: production of enemy squares
        5: production of neutral squares
    """

    window = np.zeros((window_size, window_size, 6), dtype=np.int32)

    offset = window_size // 2
    map_size_x = len(contents[0])
    map_size_y = len(contents)

    for row in range(window_size):
        for col in range(window_size):

            row_real = (y + row - offset) % map_size_y
            col_real = (x + col - offset) % map_size_x

            square = contents[row_real][col_real]

            window[row, col, 0] = square.strength if square.owner == 1 else 0
            window[row, col, 1] = square.strength if square.owner == 2 else 0
            window[row, col, 2] = square.strength if square.owner == 0 else 0

            window[row, col, 3] = square.production if square.owner == 1 else 0
            window[row, col, 4] = square.production if square.owner == 2 else 0
            window[row, col, 5] = square.production if square.owner == 0 else 0

    return window
