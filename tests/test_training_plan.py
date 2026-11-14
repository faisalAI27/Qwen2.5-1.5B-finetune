import unittest

from s1small.training_plan import dropout_seed, epoch_order, select_trial


class TrainingPlanTests(unittest.TestCase):
    def test_order_and_dropout_seeds_support_resume_and_cover_every_id(self):
        ids = ["a", "b", "c", "d", "e"]
        left = epoch_order(ids, 42, 0)
        self.assertEqual(left, epoch_order(list(reversed(ids)), 42, 0))
        self.assertCountEqual(left, ids)
        seeds = [dropout_seed(42, 0, key) for key in left]
        self.assertEqual(seeds[2:], [dropout_seed(42, 0, key) for key in left[2:]])
        self.assertNotEqual(dropout_seed(42, 0, "a"), dropout_seed(42, 1, "a"))
        with self.assertRaises(ValueError):
            epoch_order(["a", "a"], 42, 0)

    def test_selection_uses_complete_fixed_dev_loss_and_deterministic_tie(self):
        ids = ["d1", "d2"]
        base = {"status": "complete", "development_ids": ids, "development_examples": 2}
        rows = [{**base, "run_id": "higher", "learning_rate": 1e-4, "development_loss": 1.2},
                {**base, "run_id": "lower", "learning_rate": 5e-5, "development_loss": 1.2}]
        self.assertEqual(select_trial(rows, ids)["run_id"], "lower")
        rows[0]["development_loss"] = 1.1
        self.assertEqual(select_trial(rows, ids)["run_id"], "higher")
        rows[0]["development_ids"] = ["d2", "d1"]
        with self.assertRaises(ValueError):
            select_trial(rows, ids)
