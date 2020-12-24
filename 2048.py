import logic


def step(state, action):
    if(action.lower() == "w"):
        state, terminal = logic.move_up(state)

    elif(action.lower() == "s"):
        state, terminal = logic.move_down(state)

    elif(action.lower() == "a"):
        state, terminal = logic.move_left(state)

    elif(action.lower() == "d"):
        state, terminal = logic.move_right(state)

    else:
        raise ValueError(f"Invalid action: {action}")

    win = logic.check_win(state)

    if win:  # If game is won
        reward = 1
        return logic.start_game(), reward, win

    elif terminal:  # If game is lost
        reward = -1
        return logic.start_game(), reward, terminal

    else:  # Next state is valid and game is not over yet
        reward = 0
        return state, reward, terminal


if __name__ == '__main__':
    state = logic.start_game()
    print(state)

    for _ in range(3):  # Dump game loop
        state, reward, terminal = step(state, "w")
        print(state, reward, terminal)
