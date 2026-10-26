import unittest

import numpy as np

from eisfit.cuckoo import CuckooSearchConfig, cuckoo_search


class CuckooSearchTests(unittest.TestCase):
    def test_minimizes_sphere(self):
        target = np.array([0.2, 0.7, 0.45])
        result = cuckoo_search(
            lambda x: float(np.sum((x - target) ** 2)),
            dimensions=3,
            seed=42,
            config=CuckooSearchConfig(population=25, iterations=300, step_scale=0.03),
        )
        self.assertLess(result.objective, 1e-4)
        np.testing.assert_allclose(result.position, target, atol=0.02)
        self.assertTrue(all(a >= b for a, b in zip(result.history, result.history[1:])))


if __name__ == "__main__":
    unittest.main()

