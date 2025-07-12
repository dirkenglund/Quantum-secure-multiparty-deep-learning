import unittest

from Python.utils import calculate_accuracy


class DummyModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, images):
        return self.outputs.pop(0)


class TestCalculateAccuracy(unittest.TestCase):
    def test_accuracy_simple(self):
        # Two batches with two samples each
        loader = [
            (None, [0, 1]),
            (None, [0, 1])
        ]
        outputs = [
            [[10, 1], [1, 10]],  # predicts [0,1] -> 2 correct
            [[1, 10], [1, 10]]   # predicts [1,1] -> 1 correct
        ]
        model = DummyModel(outputs)
        acc = calculate_accuracy(loader, model)
        self.assertEqual(acc, 75.0)


if __name__ == '__main__':
    unittest.main()
