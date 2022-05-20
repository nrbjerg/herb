#!/usr/bin/env python3
from engine.state import State
import numpy as np


def get_string_representation(
    state: State, square_types: str = "", debug: bool = True
) -> str:
    """ Returns a string representation of the board """
    string = "   "
    for idx in range(state.dim):
        if debug:
            string += f"{idx} "
        else:
            string += f"{chr(idx + 97)} "

    for idx in range(state.dim):
        string += "\n"
        # Add the row number
        if debug:
            string += f" {idx} " if (idx <= 9) else f"{idx} "
        else:
            string += f" {chr(idx + 97)} "
        # Print add row
        for jdx in range(state.dim):
            print((idx, jdx), state.board[0][idx][jdx], state.board[1][idx][jdx])
            cell = state.board[0][idx][jdx] - state.board[1][idx][jdx]
            string += square_types[cell] + " "

    # Add column number

    return string


def main() -> None:
    white = [
        [0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0],
    ]

    black = [
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0],
    ]

    state = State(9)
    state.board = np.array([black, white])
    print(get_string_representation(state))


if __name__ == "__main__":
    main()
