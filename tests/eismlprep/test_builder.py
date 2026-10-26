import json
import unittest

from eismlprep.builder import _normalized_position


class BuilderTests(unittest.TestCase):
    def test_boundary_position_uses_parameter_scale(self):
        self.assertAlmostEqual(_normalized_position(0.1, 0.1, 1.0, "linear"), 0.0)
        self.assertAlmostEqual(_normalized_position(1.0, 0.1, 1.0, "linear"), 1.0)
        self.assertAlmostEqual(_normalized_position(1e-6, 1e-12, 1.0, "log"), 0.5)


if __name__ == "__main__":
    unittest.main()
