"""Python file to get the src path"""

import os


def get_src_path() -> str:
    """
    Get and return the src directory.

    Returns:
        The path (str) to the src directory.
    """
    return os.path.dirname(os.path.abspath(__file__))
