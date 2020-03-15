import numpy as np
import logging

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
                owned_squares.append(square)

            contents_sliced[row, col, 0] = square.strength if square.owner == 1 else 0
            contents_sliced[row, col, 1] = square.strength if square.owner == 2 else 0
            contents_sliced[row, col, 2] = square.strength if square.owner == 0 else 0

            # production of player 1, player 2 and neutral
            contents_sliced[row, col, 3] = square.production if square.owner == 1 else 0
            contents_sliced[row, col, 4] = square.production if square.owner == 2 else 0
            contents_sliced[row, col, 5] = square.production if square.owner == 0 else 0

    windows = []
    for square in owned_squares:

        half = int(window_size / 2)

        rolled = np.roll(contents_sliced, (map_size_y//2 - square.y, map_size_x//2 - square.x), (0, 1))
        #logging.debug(rolled)
        window = rolled[map_size_y//2-half : map_size_y//2 + half + 1, map_size_x//2 -half : map_size_x//2 + half + 1, : ]
        windows.append(window)

        logging.debug(window.shape)

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
    for square in owned_squares:

        half = int(window_size / 2)
        rolled = np.roll(contents_sliced, (map_size_y//2 - square.y, map_size_x//2 - square.x), (0, 1))
        #logging.debug(rolled)
        window = rolled[map_size_y//2-half : map_size_y//2 + half + 1, map_size_x//2 -half : map_size_x//2 + half + 1, : ]
        windows.append(window)

    return np.array(windows)


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
