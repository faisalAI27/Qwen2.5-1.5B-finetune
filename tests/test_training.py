import unittest

from s1small.training import accumulation_windows, supervised_count


class TrainingLogicTests(unittest.TestCase):
    def test_full_and_remainder_accumulation_windows(self):
        self.assertEqual([len(window) for window in accumulation_windows(list(range(20)), 4)], [4, 4, 4, 4, 4])
        self.assertEqual([len(window) for window in accumulation_windows(list(range(19)), 4)], [4, 4, 4, 4, 3])
        with self.assertRaises(ValueError):
            list(accumulation_windows([1], 0))

    def test_next_token_count_ignores_prompt_padding_and_dropped_first_label(self):
        self.assertEqual(supervised_count({"labels": [-100, -100, 3, 4, -100]}), 2)
        with self.assertRaises(ValueError):
            supervised_count({"labels": [-100, -100]})

    def test_token_weighted_micro_losses_sum_to_window_mean(self):
        counts, means = [2, 6, 1], [1.0, 3.0, 2.0]
        weighted = sum(mean * count / sum(counts) for mean, count in zip(means, counts))
        explicit = sum([1.0] * 2 + [3.0] * 6 + [2.0]) / 9
        self.assertAlmostEqual(weighted, explicit)


if __name__ == "__main__":
    unittest.main()
