import random

import numpy as np


def reset_game():
    mat = np.zeros((4, 4), dtype=int)
    mat[random.randint(0, 3), random.randint(0, 3)] = 2  # Fill random value with 2
    return mat


def add_new_2(mat):
    """
    Returns either:

    False: Able to place a new 2
    True: Not able to place new 2
    """
    idxs = np.argwhere(mat == 0)  # All indices where the value is 0

    if idxs.any():  # If any value is 0
        col, row = idxs[random.randint(0, idxs.shape[0]-1)]  # Random selected index
        mat[col, row] = 2
        return mat, False  # Terminal state?
    else:
        return mat, True


def check_win(mat):
    """
    Returns either:

    False: Game not over.
    True: Game won, 2048 is found in mat
    """
    if 2048 in mat:  # If won, teriminal state is needed for RL agent
        return True  # Terminal state
    else:
        return False


def compress(mat):
    new_mat = np.zeros((4, 4), dtype=int)

    for i in range(4):
        pos = 0
        for j in range(4):
            if(mat[i][j] != 0):
                new_mat[i][pos] = mat[i][j]
                pos += 1

    return new_mat


def merge(mat):
    local_reward = 0  # Collected reward during this single merge
    for i in range(4):
        for j in range(3):
            if(mat[i][j] == mat[i][j + 1] and mat[i][j] != 0):
                mat[i][j] = mat[i][j] * 2
                local_reward += mat[i][j]
                mat[i][j + 1] = 0
    return mat, local_reward


def reverse(mat):
    new_mat = mat.copy()
    new_mat = np.flip(new_mat, 1)
    return new_mat


def transpose(mat):
    new_mat = mat.copy().T
    return new_mat


def move_left(grid):
    new_grid = compress(grid)
    new_grid, reward = merge(new_grid)
    new_grid = compress(new_grid)
    new_grid, valid = add_new_2(new_grid)

    return new_grid, valid, reward


def move_right(grid):
    new_grid = reverse(grid)
    new_grid, valid, reward = move_left(new_grid)
    new_grid = reverse(new_grid)

    return new_grid, valid, reward


def move_up(grid):
    new_grid = transpose(grid)
    new_grid, valid, reward = move_left(new_grid)
    new_grid = transpose(new_grid)

    return new_grid, valid, reward


def move_down(grid):
    new_grid = transpose(grid)
    new_grid, valid, reward = move_right(new_grid)
    new_grid = transpose(new_grid)

    return new_grid, valid, reward
