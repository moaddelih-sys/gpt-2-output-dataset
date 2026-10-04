import unittest
from unittest import mock

from detector import train


class _DummyModel:
    def eval(self):
        pass


class _TqdmSpy:
    calls = []

    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = [] if iterable is None else iterable
        type(self).calls.append(kwargs)

    def __iter__(self):
        return iter(self.iterable)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def set_postfix(self, **kwargs):
        pass


class DistributedGuardTests(unittest.TestCase):
    def setUp(self):
        _TqdmSpy.calls = []

    def test_setup_distributed_cpu_path_does_not_initialize_process_group(self):
        with mock.patch.object(train.dist, "is_available", return_value=True), \
             mock.patch.object(train.torch.cuda, "is_available", return_value=False), \
             mock.patch.object(train.dist, "init_process_group") as init_process_group:
            self.assertEqual(train.setup_distributed(), (0, 1))

        init_process_group.assert_not_called()

    def test_validate_does_not_query_rank_when_process_group_is_uninitialized(self):
        with mock.patch.object(train.dist, "is_available", return_value=True), \
             mock.patch.object(train.dist, "is_initialized", return_value=False), \
             mock.patch.object(
                 train.dist,
                 "get_rank",
                 side_effect=AssertionError("get_rank must not be called without an initialized process group"),
             ) as get_rank, \
             mock.patch.object(train, "tqdm", _TqdmSpy):
            metrics = train.validate(_DummyModel(), "cpu", [])

        get_rank.assert_not_called()
        self.assertEqual([call["disable"] for call in _TqdmSpy.calls], [False, False])
        self.assertEqual(metrics, {
            "validation/accuracy": 0,
            "validation/epoch_size": 0,
            "validation/loss": 0,
        })

    def test_validate_uses_rank_when_distributed_is_initialized(self):
        with mock.patch.object(train.dist, "is_available", return_value=True), \
             mock.patch.object(train.dist, "is_initialized", return_value=True), \
             mock.patch.object(train.dist, "get_rank", return_value=1) as get_rank, \
             mock.patch.object(train, "tqdm", _TqdmSpy):
            train.validate(_DummyModel(), "cpu", [])

        self.assertEqual(get_rank.call_count, 2)
        self.assertEqual([call["disable"] for call in _TqdmSpy.calls], [True, True])

    def test_all_reduce_dict_passes_through_without_initialized_process_group(self):
        metrics = {"b": 2.0, "a": 1.0}

        with mock.patch.object(train.dist, "is_available", return_value=True), \
             mock.patch.object(train.dist, "is_initialized", return_value=False), \
             mock.patch.object(train.dist, "all_reduce") as all_reduce:
            result = train._all_reduce_dict(metrics, "cpu")

        self.assertIs(result, metrics)
        all_reduce.assert_not_called()

    def test_all_reduce_dict_keeps_collective_behavior_when_distributed_is_initialized(self):
        metrics = {"b": 2.0, "a": 1.0}

        def double_tensor(tensor):
            tensor.mul_(2)

        with mock.patch.object(train.dist, "is_available", return_value=True), \
             mock.patch.object(train.dist, "is_initialized", return_value=True), \
             mock.patch.object(train.dist, "all_reduce", side_effect=double_tensor) as all_reduce:
            result = train._all_reduce_dict(metrics, "cpu")

        self.assertEqual(result, {"a": 2.0, "b": 4.0})
        self.assertEqual(all_reduce.call_count, 2)


if __name__ == "__main__":
    unittest.main()
