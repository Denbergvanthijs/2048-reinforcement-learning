from itertools import cycle

import matplotlib.animation as animation
import matplotlib.pyplot as plt

import logic


def step(state, action):
    """
    Returns (next_state, reward, terminal) based on current (state, action) pair.
    """

    if(action.lower() == "w"):
        next_state, terminal = logic.move_up(state)

    elif(action.lower() == "a"):
        next_state, terminal = logic.move_left(state)

    elif(action.lower() == "s"):
        next_state, terminal = logic.move_down(state)

    elif(action.lower() == "d"):
        next_state, terminal = logic.move_right(state)

    else:
        raise ValueError(f"Invalid action: {action}")

    win = logic.check_win(next_state)

    if win:  # If game is won
        return logic.reset_game(), 1, win

    elif terminal:  # If a new 2 is not possible
        return logic.reset_game(), -1, terminal

    else:  # Next state is valid and game is not over yet
        return next_state, 0, terminal


if __name__ == '__main__':
    state = logic.reset_game()
    wasd_cycle = cycle("wasd")
    print(state)

    fig = plt.figure()
    ims = []
    while True:  # Dump game loop
        ims.append([plt.imshow(state, animated=True)])
        state, reward, terminal = step(state, next(wasd_cycle))
        if terminal:  # Stop at win or when no empty spot is left
            break

        print(state, reward, terminal)

    ani = animation.ArtistAnimation(fig, ims, interval=200, repeat=True, blit=True, repeat_delay=1000)
    ani.save("2048.gif")  # Can take a few seconds
    # plt.show()
