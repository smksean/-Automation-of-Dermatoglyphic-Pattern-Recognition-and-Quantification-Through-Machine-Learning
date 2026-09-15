import unittest

import numpy as np

from scripts.select_subtype_grouped_models import repeated_group_splits


class GroupedSubtypeSelectionTests(unittest.TestCase):
    def test_subjects_do_not_cross_fold_boundary(self):
        groups = np.repeat([f"S{i:02d}" for i in range(20)], 2)
        labels = np.repeat(["plain", "minority"] * 10, 2)
        features = np.arange(len(groups) * 2).reshape(len(groups), 2)

        splits = list(repeated_group_splits(features, labels, groups, n_splits=5, n_repeats=2))

        self.assertEqual(len(splits), 10)
        for _, _, _, train_idx, test_idx in splits:
            self.assertFalse(set(groups[train_idx]) & set(groups[test_idx]))
            self.assertEqual(set(labels[test_idx]), {"plain", "minority"})


if __name__ == "__main__":
    unittest.main()
