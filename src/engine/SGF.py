#!/usr/bin/env python3
import re
from engine.state import State
from tui.repressent_board import get_string_representation


def skip_uselesss_statements(sgf: str) -> str:
    """Skip useless statements """
    useless_statements = [
        re.compile(r"GN\[[\w\s]*\]"),
        re.compile(r"PB\[[\w\s]*\]"),
        re.compile(r"PW\[[\w\s]*\]"),
        re.compile(r"DT\[[\d-]*\]"),
        re.compile(r"RU\[\w*\s\]"),
        re.compile(r"C\[[\w\s*]\]"),
    ]


def from_SGF(sgf: str) -> State:
    """Produce a State, by playing the moves in the sgf format."""
    sgf_lines = sgf.splitlines()

    # Read the file info from the file
    file_info = re.match(r";FF\[(\d+)\]GM\[(\d+)\]SZ\[(\d+)\]\n", sgf)
    sgf = sgf[file_info.end() :]

    # Check that the format is correct
    if file_info.group(1) != "4":
        raise ValueError("This program only support SGF version 4")

    if file_info.group(2) != "1":
        raise ValueError("Expected game to be go in SGF file.")

    # Initalize state
    size = int(file_info.group(3))
    state = State(size)

    # Skip useless information (atleast useless to the engine )
    sgf = skip_useless_statements(sgf)
    pass


def main():
    """Load the example sgf file and print the result."""
    pass


if __name__ == "__main__":
    main()
