import logic


def step(state, action):
    if(action.lower() == "w"):
        state, _ = logic.move_up(state)
        reward = logic.get_current_state(state)

    elif(action.lower() == "s"):
        state, _ = logic.move_down(state)
        reward = logic.get_current_state(state)

    elif(action.lower() == "a"):
        state, _ = logic.move_left(state)
        reward = logic.get_current_state(state)

    elif(action.lower() == "d"):
        state, _ = logic.move_right(state)
        reward = logic.get_current_state(state)

    else:
        raise ValueError(f"Invalid action: {action}")

    if reward == 0:  # If game not over
        next_state = logic.add_new_2(state)
        return next_state, reward

    elif reward == 1:  # Win, reward is 1
        return logic.start_game(), reward

    else:  # Game over, reward is -1
        return logic.start_game(), reward


if __name__ == '__main__':
    state = logic.start_game()

    for _ in range(10):  # Dump game loop
        state, reward = step(state, "w")
        print(state, reward)
