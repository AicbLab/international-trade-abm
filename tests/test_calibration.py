"""ABC-SMC权重、目标统计及政策配对的测试契约。"""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


class CalibrationTests(unittest.TestCase):
    def test_abc_smc_reproducible_and_weighted(self):
        self.assertIsNotNone(importlib.util.find_spec('calibration'), '需要ABC-SMC模块')
        from calibration import abc_smc
        def simulate(theta, seed):
            return np.array([theta[0]])
        args = dict(simulator=simulate, bounds=np.array([[0., 1.]]), target=np.array([.7]),
                    scales=np.ones(1), particles=24, generations=3, seed=11, repeats=1)
        with tempfile.TemporaryDirectory() as tmp:
            a = abc_smc(**args, output=Path(tmp) / 'a')
            b = abc_smc(**args, output=Path(tmp) / 'b')
        self.assertTrue(a['complete'])
        np.testing.assert_array_equal(a['particles'], b['particles'])
        self.assertAlmostEqual(a['weights'].sum(), 1.)
        self.assertLess(abs(np.average(a['particles'][:, 0], weights=a['weights']) - .7), .1)
        eps = [g['epsilon'] for g in a['diagnostics']]
        self.assertTrue(all(x >= y for x, y in zip(eps, eps[1:])))
        self.assertTrue(all(1 <= g['ess'] <= 24.00001 for g in a['diagnostics']))

    def test_importance_weights_use_previous_mixture(self):
        self.assertIsNotNone(importlib.util.find_spec('calibration'), '需要ABC-SMC模块')
        from calibration import importance_weights
        from scipy.stats import norm
        previous = np.array([[.2], [.8]])
        candidates = np.array([[.25], [.6], [.85]])
        old_weights = np.array([.9, .1])
        expected = 1 / (norm.pdf(candidates, previous.T, .15) @ old_weights)
        expected /= expected.sum()
        np.testing.assert_allclose(importance_weights(candidates, previous, old_weights,
                                                     np.array([.15])), expected)

    def test_invalid_scale_rejected(self):
        self.assertIsNotNone(importlib.util.find_spec('calibration'), '需要ABC-SMC模块')
        from calibration import abc_smc
        with self.assertRaises(ValueError):
            abc_smc(lambda t, s: t, np.array([[0., 1.]]), np.array([.5]),
                    np.array([0.]), particles=4, generations=1)

    def test_exhausted_generation_not_marked_complete(self):
        self.assertIsNotNone(importlib.util.find_spec('calibration'), '需要ABC-SMC模块')
        from calibration import abc_smc
        result = abc_smc(lambda t, s: t, np.array([[0., 1.]]), np.array([.5]),
                         np.ones(1), particles=4, generations=1,
                         max_attempts=1, repeats=1)
        self.assertFalse(result['complete'])
        self.assertEqual(result['diagnostics'][0]['attempts'], 1)

    def test_training_summary_ignores_holdout(self):
        from research import summary_vector
        states = np.ones((24, 4, 5), dtype=np.uint8)
        before = summary_vector(states)
        states[16:] = 0
        np.testing.assert_array_equal(before, summary_vector(states))

    def test_eci_finite_and_scale_invariant(self):
        self.assertIsNotNone(importlib.util.find_spec('research'), '需要统一实验上下文')
        from research import regional_eci
        m = np.array([[1,1,1,0], [1,1,0,0], [0,1,0,1], [0,0,1,1]])
        eci = regional_eci(m)
        self.assertTrue(np.isfinite(eci).all())
        self.assertAlmostEqual(eci.mean(), 0.)
        self.assertAlmostEqual(eci.std(), 1.)
        np.testing.assert_allclose(eci, regional_eci(np.tile(m, (1, 2))), atol=1e-10)
        self.assertTrue(np.isfinite(regional_eci(np.zeros((4,4)))).all())


if __name__ == '__main__':
    unittest.main()
