import unittest

from src.fraud_detection.schema import FEATURE_COLUMNS, NOTEBOOK_SELECTED_FEATURES, TARGET_COLUMN


class SchemaTests(unittest.TestCase):
    def test_kaggle_feature_schema(self):
        self.assertEqual(len(FEATURE_COLUMNS), 30)
        self.assertEqual(FEATURE_COLUMNS[0], "Time")
        self.assertEqual(FEATURE_COLUMNS[-1], "Amount")
        self.assertEqual(TARGET_COLUMN, "Class")

    def test_notebook_selected_features_are_valid(self):
        missing = set(NOTEBOOK_SELECTED_FEATURES).difference(FEATURE_COLUMNS)
        self.assertEqual(missing, set())


if __name__ == "__main__":
    unittest.main()
