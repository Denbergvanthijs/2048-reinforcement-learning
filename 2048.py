import logic

if __name__ == '__main__':
    mat = logic.start_game()

while(True):
    print(mat)
    x = input("Press the command : ")

    if(x.lower() == "w"):
        mat, flag = logic.move_up(mat)
        status = logic.get_current_state(mat)

        if(status == 0):  # If game not over
            logic.add_new_2(mat)
        else:
            break

    elif(x.lower() == "s"):
        mat, flag = logic.move_down(mat)
        status = logic.get_current_state(mat)

        if(status == 0):  # If game not over
            logic.add_new_2(mat)
        else:
            break

    elif(x.lower() == "a"):
        mat, flag = logic.move_left(mat)
        status = logic.get_current_state(mat)

        if(status == 0):  # If game not over
            logic.add_new_2(mat)
        else:
            break

    elif(x.lower() == "d"):
        mat, flag = logic.move_right(mat)
        status = logic.get_current_state(mat)

        if(status == 0):  # If game not over
            logic.add_new_2(mat)
        else:
            break

    elif(x.lower() == "q"):
        exit()
    else:
        print("Invalid Key Pressed")

    print(mat)
