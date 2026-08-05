from pathlib import Path
import sys
import unittest

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.td3.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    @staticmethod
    def make_buffer(**overrides):
        parameters = {
            "state_dim": 3,
            "action_dim": 2,
            "max_size": 8,
            "seed": 123,
            "device": "cpu",
        }
        parameters.update(overrides)
        return ReplayBuffer(**parameters)

    @staticmethod
    def add_transition(buffer, value, terminated=False):
        buffer.add(
            state=np.array([value, value + 1, value + 2]),
            action=np.array([value / 10, -value / 10]),
            next_state=np.array([value + 3, value + 4, value + 5]),
            reward=float(value),
            terminated=terminated,
        )

    def test_terminal_masks_and_terminal_next_state_are_stored(self):
        buffer = self.make_buffer()
        terminal_next_state = np.array([9.0, 8.0, 7.0])

        buffer.add(
            state=np.array([1.0, 2.0, 3.0]),
            action=np.array([0.1, -0.1]),
            next_state=np.array([4.0, 5.0, 6.0]),
            reward=1.5,
            terminated=False,
        )
        buffer.add(
            state=np.array([3.0, 2.0, 1.0]),
            action=np.array([-0.2, 0.2]),
            next_state=terminal_next_state,
            reward=-2.0,
            terminated=True,
        )

        self.assertEqual(len(buffer), 2)
        np.testing.assert_array_equal(
            buffer.bootstrap_mask[:2, 0],
            np.array([1.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            buffer.next_state[1],
            terminal_next_state.astype(np.float32),
        )

    def test_storage_and_sample_shapes_dtype_and_cpu_device(self):
        buffer = self.make_buffer()
        for value in range(4):
            self.add_transition(buffer, value)

        for storage in (
            buffer.state,
            buffer.action,
            buffer.next_state,
            buffer.reward,
            buffer.bootstrap_mask,
        ):
            self.assertIsInstance(storage, np.ndarray)
            self.assertEqual(storage.dtype, np.float32)

        sample = buffer.sample(batch_size=5)
        expected_shapes = ((5, 3), (5, 2), (5, 3), (5, 1), (5, 1))
        for tensor, expected_shape in zip(sample, expected_shapes):
            self.assertEqual(tensor.shape, expected_shape)
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertEqual(tensor.device, torch.device("cpu"))
            self.assertTrue(torch.isfinite(tensor).all().item())

    def test_same_seed_produces_same_samples(self):
        first = self.make_buffer(seed=987)
        second = self.make_buffer(seed=987)
        for value in range(8):
            self.add_transition(first, value, terminated=value % 3 == 0)
            self.add_transition(second, value, terminated=value % 3 == 0)

        first_sample = first.sample(batch_size=12)
        second_sample = second.sample(batch_size=12)

        for first_tensor, second_tensor in zip(first_sample, second_sample):
            self.assertTrue(torch.equal(first_tensor, second_tensor))

    def test_ring_buffer_overwrites_oldest_transitions(self):
        buffer = self.make_buffer(max_size=3)
        for value in range(5):
            self.add_transition(buffer, value)

        self.assertEqual(len(buffer), 3)
        self.assertEqual(buffer.ptr, 2)
        np.testing.assert_array_equal(
            buffer.state[:, 0],
            np.array([3.0, 4.0, 2.0], dtype=np.float32),
        )

    def test_invalid_add_inputs_do_not_mutate_buffer(self):
        buffer = self.make_buffer()
        self.add_transition(buffer, 1)
        size_before = len(buffer)
        ptr_before = buffer.ptr
        storage_before = tuple(
            storage.copy()
            for storage in (
                buffer.state,
                buffer.action,
                buffer.next_state,
                buffer.reward,
                buffer.bootstrap_mask,
            )
        )

        valid_state = np.array([1.0, 2.0, 3.0])
        valid_action = np.array([0.1, -0.1])
        valid_next_state = np.array([4.0, 5.0, 6.0])
        cases = (
            {"state": np.zeros((1, 3))},
            {"action": np.zeros((2, 1))},
            {"next_state": np.zeros(2)},
            {"reward": np.array([1.0])},
            {"state": np.array([1.0, np.nan, 3.0])},
            {"action": np.array([np.inf, 0.0])},
            {"next_state": np.array([1.0, 2.0, -np.inf])},
            {"reward": np.nan},
            {"terminated": 1},
        )

        for override in cases:
            with self.subTest(override=override):
                transition = {
                    "state": valid_state,
                    "action": valid_action,
                    "next_state": valid_next_state,
                    "reward": 1.0,
                    "terminated": False,
                }
                transition.update(override)

                with self.assertRaises(ValueError):
                    buffer.add(**transition)

                self.assertEqual(len(buffer), size_before)
                self.assertEqual(buffer.ptr, ptr_before)
                for storage, expected_storage in zip(
                    (
                        buffer.state,
                        buffer.action,
                        buffer.next_state,
                        buffer.reward,
                        buffer.bootstrap_mask,
                    ),
                    storage_before,
                ):
                    np.testing.assert_array_equal(
                        storage,
                        expected_storage,
                    )

    def test_rejects_invalid_dimensions_capacity_and_empty_sampling(self):
        for name in ("state_dim", "action_dim", "max_size"):
            for value in (0, -1, 2.0, True):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        self.make_buffer(**{name: value})

        buffer = self.make_buffer()
        with self.assertRaises(ValueError):
            buffer.sample(batch_size=1)

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is not available",
    )
    def test_cuda_sampling_returns_cuda_tensors(self):
        buffer = self.make_buffer(device="cuda:0")
        self.add_transition(buffer, 1)

        sample = buffer.sample(batch_size=2)

        for tensor in sample:
            self.assertEqual(tensor.device, torch.device("cuda:0"))
            self.assertEqual(tensor.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
