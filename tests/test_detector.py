from __future__ import annotations

import unittest

import numpy as np

from detecteur_image_ia.detector import features_from_vectors


class DetectorTests(unittest.TestCase):
    def test_covariance_features_match_flattened_vectors(self) -> None:
        vx = np.array([[0, 1_000_000], [2_000_000, 3_000_000]], dtype=np.int32)
        vy = np.array([[0, 2_000_000], [4_000_000, 6_000_000]], dtype=np.int32)

        features = features_from_vectors(vx, vy)

        cloud = np.column_stack([vx.ravel(), vy.ravel()]).astype(np.float64) / 1_000_000.0
        centered = cloud - cloud.mean(axis=0)
        covariance = centered.T @ centered / cloud.shape[0]

        self.assertAlmostEqual(features["mean_x"], float(cloud[:, 0].mean()))
        self.assertAlmostEqual(features["mean_y"], float(cloud[:, 1].mean()))
        self.assertAlmostEqual(features["cov_xx"], float(covariance[0, 0]))
        self.assertAlmostEqual(features["cov_xy"], float(covariance[0, 1]))
        self.assertAlmostEqual(features["cov_yy"], float(covariance[1, 1]))


if __name__ == "__main__":
    unittest.main()

