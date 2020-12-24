import random

import numpy as np


def start_game():
    mat = np.zeros((4, 4), dtype=int)
    mat[random.randint(0, 3), random.randint(0, 3)] = 2
    return mat


def add_new_2(mat):
    idxs = np.argwhere(mat == 0)  # All indices where the value is 0
    if idxs.any():
        col, row = idxs[random.randint(0, idxs.shape[0]-1)]  # Random selected index
        mat[col, row] = 2
        return mat
    else:
        return 0


def get_current_state(mat):
    """
    Returns either:

    0: Game not over.
    1: Game won, 2048 is found in mat
    -1: Game is lost, a new 2 can not be placed
    """
    if 2048 in mat:  # If won, teriminal state is needed for RL agent
        return 1
    idxs = np.argwhere(mat == 0)

    if idxs.any():  # If any value is 0, thus a new 2 can be placed
        return 0
    else:  # Game over if no new 2 can be placed
        return -1


def compress(mat):
    changed = False
    new_mat = np.zeros((4, 4), dtype=int)

    for i in range(4):
        pos = 0
        for j in range(4):
            if(mat[i][j] != 0):
                new_mat[i][pos] = mat[i][j]

                if(j != pos):
                    changed = True
                pos += 1

    return new_mat, changed


def merge(mat):
    changed = False

    for i in range(4):
        for j in range(3):
            if(mat[i][j] == mat[i][j + 1] and mat[i][j] != 0):
                mat[i][j] = mat[i][j] * 2
                mat[i][j + 1] = 0

                changed = True

    return mat, changed


def reverse(mat):
    new_mat = mat.copy()
    new_mat = np.flip(new_mat, 1)
    return new_mat


def transpose(mat):
    new_mat = mat.copy().T
    return new_mat


def move_left(grid):
    new_grid, changed1 = compress(grid)
    new_grid, changed2 = merge(new_grid)
    new_grid, _ = compress(new_grid)

    changed = changed1 or changed2
    return new_grid, changed


def move_right(grid):
    new_grid = reverse(grid)
    new_grid, changed = move_left(new_grid)
    new_grid = reverse(new_grid)
    return new_grid, changed


def move_up(grid):
    new_grid = transpose(grid)
    new_grid, changed = move_left(new_grid)
    new_grid = transpose(new_grid)
    return new_grid, changed


def move_down(grid):
    new_grid = transpose(grid)
    new_grid, changed = move_right(new_grid)
    new_grid = transpose(new_grid)
    return new_grid, changed


if __name__ == "__main__":
    mat = start_game()
    # mat = np.zeros((4, 4), dtype=int)
    # mat[0] = 1
    print(mat)
    mat, state = move_left(mat)
    print(mat, state)
    mat = add_new_2(mat)
    print(mat)
    mat, state = move_right(mat)
    print(mat, state)

    # mat = np.ones(shape=(4, 4))  # Game over
    # res = get_current_state(mat)
    # print(res)
    # mat = np.full((4, 4), 2048)  # Win
    # res = get_current_state(mat)
    # print(res)
