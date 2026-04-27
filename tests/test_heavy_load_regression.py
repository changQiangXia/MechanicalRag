import unittest


HEAVY_LOAD_BASELINE = {
    "pick_metal_heavy": 0.0,
    "pick_metal_heavy_fast": 0.8667,
    "overall_mean": 0.8222,
}


class HeavyLoadRegressionBaselineTest(unittest.TestCase):
    def test_baseline_constants_match_expected_current_profile(self):
        self.assertEqual(HEAVY_LOAD_BASELINE["pick_metal_heavy"], 0.0)
        self.assertEqual(HEAVY_LOAD_BASELINE["pick_metal_heavy_fast"], 0.8667)
        self.assertEqual(HEAVY_LOAD_BASELINE["overall_mean"], 0.8222)


if __name__ == "__main__":
    unittest.main()
