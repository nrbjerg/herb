#!/usr/bin/env python3
from engine.state import State


def get_string_representation(state: State) -> str:
    """ Returns a string representation of the board """
    string = ""

    for idx in range(state.dim):
        # Add the row number
        string += (
            f"{state.dim - idx} " if (state.dim - idx) >= 10 else f" {state.dim - idx} "
        )
        # Print add row
        for jdx in range(state.dim):
            cell = state.board[0][idx][jdx] - state.board[1][idx][jdx]
            string += square_types[cell] + " "
        string += "\n"

    # Add column number
    string += "   "
    for idx in range(state.dim):
        string += f"{chr(idx + 97)} "

    return string
