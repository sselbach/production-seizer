import numpy as np

def get_window_at(contents, x, y, window_size=7):
    """Given the map in halite format, returns a window of `window_size` centered at `x` and `y`.
    The window is a np.ndarray of shape (window_size, window_size, 6), where the last dimension contains the
    different input channels:

        0: strength of units of player 1
        1: strength of units of player 2
        2: strength of neutral units
        3: production of player 1 squares
        4: production of player 2 squares
        5: production of neutral squares
    """

    window = np.zeros((window_size, window_size, 6), dtype=np.int32)

    offset = window_size // 2
    map_size_x = len(contents[0])
    map_size_y = len(contents)

    for row in range(window_size):
        for col in range(window_size):

            # compute position of the currently looked at square
            row_real = (y + row - offset) % map_size_y
            col_real = (x + col - offset) % map_size_x

            square = contents[row_real][col_real]

            # strength of player 1, player 2 and neutral
            window[row, col, 0] = square.strength if square.owner == 1 else 0
            window[row, col, 1] = square.strength if square.owner == 2 else 0
            window[row, col, 2] = square.strength if square.owner == 0 else 0

            # production of player 1, player 2 and neutral
            window[row, col, 3] = square.production if square.owner == 1 else 0
            window[row, col, 4] = square.production if square.owner == 2 else 0
            window[row, col, 5] = square.production if square.owner == 0 else 0

    return window


def convert_map_to_numpy(contents, include_owner_channel=False):
    """Converts the entire map into a 'window', except there may not be a center square"""

    map_size_y = len(contents)
    map_size_x = len(contents[0])

    window = np.zeros((map_size_y, map_size_x, 7 if include_owner_channel else 6))

    for row in range(map_size_y):
        for col in range(map_size_x):
            square = contents[row][col]

            # strength of player 1, player 2 and neutral
            window[row, col, 0] = square.strength if square.owner == 1 else 0
            window[row, col, 1] = square.strength if square.owner == 2 else 0
            window[row, col, 2] = square.strength if square.owner == 0 else 0

            # production of player 1, player 2 and neutral
            window[row, col, 3] = square.production if square.owner == 1 else 0
            window[row, col, 4] = square.production if square.owner == 2 else 0
            window[row, col, 5] = square.production if square.owner == 0 else 0

            if include_owner_channel:
                window[row, col, 6] = square.owner

    return window


def get_windows(contents, window_size = 7):
    """
    Returns list of windows for each square owned by the Bot
    """
    map_size_x = len(contents[0])
    map_size_y = len(contents)

    contents_sliced = np.zeros((map_size_y, map_size_x, 6))

    owned_squares = []

    for row in range(map_size_y):
        for col in range(map_size_x):

            square = contents[row][col]

            if(square.owner == 1):
                owned_squares.append((square.x, square.y))

            contents_sliced[row, col, 0] = square.strength if square.owner == 1 else 0
            contents_sliced[row, col, 1] = square.strength if square.owner == 2 else 0
            contents_sliced[row, col, 2] = square.strength if square.owner == 0 else 0

            # production of player 1, player 2 and neutral
            contents_sliced[row, col, 3] = square.production if square.owner == 1 else 0
            contents_sliced[row, col, 4] = square.production if square.owner == 2 else 0
            contents_sliced[row, col, 5] = square.production if square.owner == 0 else 0

    windows = []
    for x, y in owned_squares:

        half = int(window_size / 2)

        window = contents_sliced[y - half : y + half + 1, x - half : x + half + 1, : ]
        windows.append(window)

    return owned_squares, np.array(windows)


def get_windows_for_squares(contents, owned_squares, window_size = 7):
    map_size_x = len(contents[0])
    map_size_y = len(contents)

    contents_sliced = np.zeros((map_size_y, map_size_x, 6))

    for row in range(map_size_y):
        for col in range(map_size_x):

            square = contents[row][col]

            contents_sliced[row, col, 0] = square.strength if square.owner == 1 else 0
            contents_sliced[row, col, 1] = square.strength if square.owner == 2 else 0
            contents_sliced[row, col, 2] = square.strength if square.owner == 0 else 0

            # production of player 1, player 2 and neutral
            contents_sliced[row, col, 3] = square.production if square.owner == 1 else 0
            contents_sliced[row, col, 4] = square.production if square.owner == 2 else 0
            contents_sliced[row, col, 5] = square.production if square.owner == 0 else 0

    windows = []
    for x, y in owned_squares:

        half = int(window_size / 2)

        window = contents_sliced[y - half : y + half + 1, x - half : x + half + 1, : ]
        windows.append(window)

    return owned_squares, np.array(windows)
