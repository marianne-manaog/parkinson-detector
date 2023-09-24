"""Test to ensure src dir is retrieved correctly"""

import os
import unittest

from src.get_src_dir import get_src_path


class TestGetSrcDir(unittest.TestCase):
    """
    Test class to ensure src dir is retrieved correctly
    """

    def test_get_src_path(self):
        """
        Unit test to ensure src dir is retrieved correctly
        """
        result_path = get_src_path().split(os.sep)[-1]
        expected_path = 'src'
        self.assertEqual(result_path, expected_path)
