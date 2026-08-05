from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.td3.td3 import TD3


class TestTD3Device(unittest.TestCase):
    @staticmethod
    def make_td3(device):
        return TD3(
            state_dim=3,
            action_dim=2,
            max_action=1.0,
            device=device,
        )

    def assert_networks_on_device(self, policy, expected_device):
        for name in (
            "actor",
            "actor_target",
            "critic",
            "critic_target",
        ):
            network = getattr(policy, name)
            parameter_devices = {
                parameter.device for parameter in network.parameters()
            }
            self.assertEqual(
                parameter_devices,
                {expected_device},
                msg=f"{name} is not on {expected_device}",
            )

    def test_explicit_cpu_and_torch_device(self):
        for requested_device in ("cpu", torch.device("cpu")):
            with self.subTest(requested_device=requested_device):
                policy = self.make_td3(requested_device)

                self.assertEqual(policy.device, torch.device("cpu"))
                self.assert_networks_on_device(
                    policy,
                    torch.device("cpu"),
                )

    def test_auto_and_none_select_available_device(self):
        expected_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        for requested_device in (None, "auto"):
            with self.subTest(requested_device=requested_device):
                policy = self.make_td3(requested_device)

                self.assertEqual(policy.device, expected_device)
                self.assert_networks_on_device(policy, expected_device)

    def test_supported_cuda_strings_are_parsed(self):
        for requested_device in ("cuda", "cuda:0", "cuda:1"):
            with self.subTest(requested_device=requested_device):
                self.assertEqual(
                    TD3._resolve_device(requested_device),
                    torch.device(requested_device),
                )

    def test_cpu_select_action_returns_finite_numpy_vector(self):
        policy = self.make_td3("cpu")
        state = np.array([0.25, -0.5, 1.0], dtype=np.float32)

        action = policy.select_action(state)

        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape, (2,))
        self.assertTrue(np.all(np.isfinite(action)))

    def test_cpu_save_load_preserves_action(self):
        policy = self.make_td3("cpu")
        state = np.array([0.25, -0.5, 1.0], dtype=np.float32)
        expected_action = policy.select_action(state)

        with tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT
        ) as temporary_directory:
            checkpoint = str(Path(temporary_directory) / "td3")
            policy.save(checkpoint)

            restored_policy = self.make_td3(torch.device("cpu"))
            restored_policy.load(checkpoint)
            restored_action = restored_policy.select_action(state)

        np.testing.assert_array_equal(restored_action, expected_action)
        self.assert_networks_on_device(
            restored_policy,
            torch.device("cpu"),
        )

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is not available",
    )
    def test_explicit_cuda_places_all_networks_on_cuda(self):
        policy = self.make_td3("cuda:0")

        self.assertEqual(policy.device, torch.device("cuda:0"))
        self.assert_networks_on_device(
            policy,
            torch.device("cuda:0"),
        )


if __name__ == "__main__":
    unittest.main()
