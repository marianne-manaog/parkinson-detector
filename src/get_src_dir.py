import os


def get_src_path() -> str:
    """
    Get and return the src directory.

    Returns:
        The path to the src directory.
    """
    return os.path.dirname(os.path.abspath(__file__))

