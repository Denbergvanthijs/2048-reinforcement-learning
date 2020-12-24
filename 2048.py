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
    print(state)

    for _ in range(3):  # Dump game loop
        state, reward, terminal = step(state, "w")
        print(state, reward, terminal)
