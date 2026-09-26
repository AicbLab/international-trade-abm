"""计算口径、随机复现与政策作用域的回归测试。"""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pipeline import build_proximity, build_matrices, clean, resource_product_mask
from model import ComplexityABM, ProductSpace, PolicyConfig


def toy_model(**kwargs):
    p = [f'{i:04d}' for i in range(40)]
    phi = np.ones((40, 40)) - np.eye(40)
    caps = {'CHN': np.r_[np.ones(10), np.zeros(30)], 'JPN': np.r_[np.zeros(10), np.ones(30)]}
    defaults = dict(product_space=ProductSpace(p, phi), initial_capabilities=caps,
                    learning_rate=0.5, imitation_rate=0.5, investment_budget=2,
                    depreciation_rate=0.1, seed=17)
    defaults.update(kwargs)
    return ComplexityABM(**defaults)


class PipelineTests(unittest.TestCase):
    def test_proximity_symmetric_and_exact(self):
        m = np.array([[1, 1], [1, 0], [1, 0]], dtype=np.uint8)
        np.testing.assert_allclose(build_proximity(m), [[0, 1/3], [1/3, 0]])

    def test_proximity_no_uint8_overflow(self):
        m = np.ones((300, 2), dtype=np.uint8)
        self.assertEqual(build_proximity(m)[0, 1], 1)

    def test_resource_proxy_not_chemicals_or_cotton(self):
        np.testing.assert_array_equal(resource_product_mask(['2601', '2709', '2901', '5201', '8542']),
                                      [True, True, False, False, False])

    def test_duplicate_keys_rejected(self):
        df = pd.DataFrame([['CHN', '0101', 2000, 1.0]] * 2,
                          columns=['country_iso3_code', 'product_hs92_code', 'year', 'export_value'])
        with self.assertRaises(ValueError):
            clean(df)

    def test_global_reference_and_base_year(self):
        rows = [['CHN', '0101', 2000, 90.], ['CHN', '0102', 2000, 10.],
                ['USA', '0101', 2000, 10.], ['USA', '0102', 2000, 90.]]
        df = pd.DataFrame(rows, columns=['country_iso3_code', 'product_hs92_code', 'year', 'export_value'])
        a = build_matrices(df)
        np.testing.assert_allclose(a['rca'][0, :, 0], [1.8, .2])
        self.assertIn('M_base_global', a)
        future = df.assign(year=2023, export_value=[1., 999., 999., 1.])
        b = build_matrices(pd.concat([df, future], ignore_index=True))
        np.testing.assert_array_equal(a['M_base_global'], b['M_base_global'])


class ModelTests(unittest.TestCase):
    def test_reproducible_seed(self):
        a, b = toy_model(), toy_model()
        for _ in range(8):
            a.step(); b.step()
        pd.testing.assert_frame_equal(a.history(), b.history())

    def test_initial_state_recorded(self):
        model = toy_model()
        self.assertEqual(model.history()['step'].tolist(), [0])

    def test_symmetric_sparse_graph(self):
        model = toy_model()
        a = model.product_space.adj
        np.testing.assert_array_equal(a, a.T)
        self.assertEqual(model.product_space.graph.number_of_edges(), int(a.sum() // 2))

    def test_budget_limits_successes(self):
        model = toy_model(learning_rate=1., imitation_rate=0., depreciation_rate=0.)
        before = [int(a.capabilities.sum()) for a in model.agents]
        model.step()
        for agent, old in zip(model.agents, before):
            self.assertLessEqual(int(agent.capabilities.sum()) - old, 2)

    def test_initial_capabilities_can_be_lost(self):
        model = toy_model(learning_rate=0., imitation_rate=0., depreciation_rate=1.)
        model.step()
        self.assertEqual(sum(a.capabilities.sum() for a in model.agents), 0)

    def test_resource_mask_also_blocks_imitation(self):
        mask = np.ones(40, dtype=bool)
        model = toy_model(resource_mask=mask, learning_rate=1., imitation_rate=1., depreciation_rate=0.)
        initial = [a.capabilities.copy() for a in model.agents]
        for _ in range(20):
            model.step()
        for agent, old in zip(model.agents, initial):
            np.testing.assert_array_equal(agent.capabilities, old)

    def test_policy_scope_and_start(self):
        self.assertIn('countries', PolicyConfig.__dataclass_fields__)
        cfg = PolicyConfig(countries=['CHN'], start_step=2, blockade_targets=['0000'], blockade_strength=1.)
        model = toy_model(learning_rate=0., imitation_rate=0., depreciation_rate=0., policy_config=cfg)
        model.step()
        self.assertEqual(list(model.agents)[0].capabilities[0], 1)
        model.step()
        self.assertEqual(list(model.agents)[0].capabilities[0], 0)
        self.assertEqual(list(model.agents)[1].capabilities.sum(), 30)

    def test_invalid_budget_rejected(self):
        with self.assertRaises(ValueError):
            toy_model(investment_budget=-1.)

    def test_heterogeneous_budget(self):
        budgets = {'CHN': 10., 'JPN': 1.}
        model = toy_model(investment_budget=budgets, learning_rate=1.,
                          imitation_rate=0., depreciation_rate=0.)
        model.step()
        agents = list(model.agents)
        # CHN has budget 10 → up to 10 new capabilities; JPN budget 1 → up to 1
        # But CHN already has products 0-9, JPN has 10-39.
        # Eligible = not already owned AND investable AND exposed.
        # With uniform phi, all products are neighbors.
        # CHN can gain at most min(10, 30) = 10 new (products 10-39 are eligible)
        # JPN can gain at most min(1, 10) = 1 new (products 0-9 are eligible)
        chn_gain = int(agents[0].capabilities.sum()) - 10  # CHN started with 10
        jpn_gain = int(agents[1].capabilities.sum()) - 30  # JPN started with 30
        self.assertLessEqual(chn_gain, 10)
        self.assertLessEqual(jpn_gain, 1)

    def test_tariff_reduces_investment_multiplier(self):
        from model import PolicyModule, PolicyConfig
        products = ['0000', '0001', '0002']
        cfg = PolicyConfig(tariff_targets=['0001'], tariff_rate=0.5,
                           tariff_elasticity=1.0)
        pm = PolicyModule(products, cfg)
        mult = pm.multiplier()
        self.assertAlmostEqual(mult[0], 1.0)  # untarged
        self.assertLess(mult[1], 1.0)  # tariff reduces multiplier
        self.assertAlmostEqual(mult[1], (1.5) ** (-1.0))
        self.assertAlmostEqual(mult[2], 1.0)  # untarged

    def test_reproducibility_with_policy(self):
        cfg = PolicyConfig(subsidy_targets=['0000'], subsidy_strength=2.0,
                           countries=['CHN'], start_step=1)
        a = toy_model(policy_config=cfg)
        b = toy_model(policy_config=cfg)
        for _ in range(10):
            a.step(); b.step()
        pd.testing.assert_frame_equal(a.history(), b.history())


if __name__ == '__main__':
    unittest.main()
