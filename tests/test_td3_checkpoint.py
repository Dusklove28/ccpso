from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.td3.replay_buffer import ReplayBuffer
from agents.td3.td3 import TD3


class TestTD3Checkpoint(unittest.TestCase):
    @staticmethod
    def make_policy(device="cpu", **overrides):
        parameters = {
            "state_dim": 3,
            "action_dim": 1,
            "max_action": 1.0,
            "discount": 0.97,
            "tau": 0.01,
            "policy_noise": 0.15,
            "noise_clip": 0.4,
            "policy_freq": 2,
            "device": device,
        }
        parameters.update(overrides)
        return TD3(**parameters)

    @staticmethod
    def make_trained_policy(device="cpu"):
        torch.manual_seed(123)
        policy = TestTD3Checkpoint.make_policy(device=device)
        replay_buffer = ReplayBuffer(
            state_dim=3,
            action_dim=1,
            max_size=16,
            seed=456,
            device=device,
        )
        for index in range(8):
            value = float(index + 1)
            replay_buffer.add(
                state=np.array([value, -value, value / 2]),
                action=np.array([(-1.0) ** index * 0.25]),
                next_state=np.array(
                    [value + 0.1, -value + 0.2, value / 2 + 0.3]
                ),
                reward=value / 10,
                terminated=index in (3, 7),
            )
        for _ in range(4):
            policy.train(replay_buffer, batch_size=4)
        return policy

    def assert_networks_equal(self, expected, actual):
        for network_name in (
            "actor",
            "actor_target",
            "critic",
            "critic_target",
        ):
            expected_state = getattr(expected, network_name).state_dict()
            actual_state = getattr(actual, network_name).state_dict()
            self.assertEqual(set(expected_state), set(actual_state))
            for name in expected_state:
                self.assertTrue(
                    torch.equal(
                        expected_state[name].detach().cpu(),
                        actual_state[name].detach().cpu(),
                    ),
                    msg=f"{network_name}.{name} differs",
                )

    def assert_networks_on_device(self, policy, device):
        for network_name in (
            "actor",
            "actor_target",
            "critic",
            "critic_target",
        ):
            devices = {
                parameter.device
                for parameter in getattr(
                    policy,
                    network_name,
                ).parameters()
            }
            self.assertEqual(devices, {torch.device(device)})

    def assert_nested_equal(self, expected, actual):
        if torch.is_tensor(expected):
            self.assertTrue(
                torch.equal(expected.detach().cpu(), actual.detach().cpu())
            )
        elif isinstance(expected, dict):
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                self.assert_nested_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual))
            for expected_item, actual_item in zip(expected, actual):
                self.assert_nested_equal(expected_item, actual_item)
        else:
            self.assertEqual(expected, actual)

    def test_model_only_round_trip_restores_networks_and_metadata(self):
        source = self.make_trained_policy()
        state = np.array([0.25, -0.5, 0.75], dtype=np.float32)
        expected_action = source.select_action(state)
        metadata = {
            "problem": "Sphere",
            "dimensions": 3,
            "training_seed": 123,
            "tags": ["td3", "evaluation"],
        }

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            path = Path(temp_dir) / "policy.pt"
            saved_path = source.save_checkpoint(path, metadata=metadata)
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(
                set(checkpoint),
                {
                    "format_version",
                    "state_dim",
                    "action_dim",
                    "max_action",
                    "hyperparameters",
                    "actor",
                    "actor_target",
                    "critic",
                    "critic_target",
                    "actor_optimizer",
                    "critic_optimizer",
                    "total_it",
                    "metadata",
                },
            )
            self.assertEqual(
                checkpoint["format_version"],
                TD3.CHECKPOINT_FORMAT_VERSION,
            )
            self.assertNotIn("replay_buffer", checkpoint)
            self.assertNotIn("rng", checkpoint)
            restored = self.make_policy(
                max_action=2.0,
                discount=0.5,
                tau=0.1,
                policy_noise=0.3,
                noise_clip=0.2,
                policy_freq=5,
            )

            loaded_metadata = restored.load_checkpoint(path)

            self.assertEqual(saved_path, str(path.resolve()))
            self.assertEqual(loaded_metadata, metadata)
            self.assertEqual(restored.total_it, source.total_it)
            self.assertEqual(restored.max_action, source.max_action)
            self.assertEqual(restored.discount, source.discount)
            self.assertEqual(restored.tau, source.tau)
            self.assertEqual(restored.policy_noise, source.policy_noise)
            self.assertEqual(restored.noise_clip, source.noise_clip)
            self.assertEqual(restored.policy_freq, source.policy_freq)
            self.assertEqual(restored.actor_optimizer.state, {})
            self.assertEqual(restored.critic_optimizer.state, {})
            for network_name in (
                "actor",
                "actor_target",
                "critic",
                "critic_target",
            ):
                network = getattr(restored, network_name)
                self.assertIs(network.training, False)
                self.assertTrue(
                    all(
                        not parameter.requires_grad
                        for parameter in network.parameters()
                    )
                )
            self.assert_networks_equal(source, restored)
            self.assert_networks_on_device(restored, "cpu")
            np.testing.assert_array_equal(
                restored.select_action(state),
                expected_action,
            )

    def test_optimizer_state_can_be_restored(self):
        source = self.make_trained_policy()

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            path = Path(temp_dir) / "policy.pt"
            source.save_checkpoint(path)
            restored = self.make_policy()

            restored.load_checkpoint(path, load_optimizers=True)

        self.assert_nested_equal(
            source.actor_optimizer.state_dict(),
            restored.actor_optimizer.state_dict(),
        )
        self.assert_nested_equal(
            source.critic_optimizer.state_dict(),
            restored.critic_optimizer.state_dict(),
        )
        for network_name in (
            "actor",
            "actor_target",
            "critic",
            "critic_target",
        ):
            network = getattr(restored, network_name)
            self.assertIs(network.training, True)
            self.assertTrue(
                all(
                    parameter.requires_grad
                    for parameter in network.parameters()
                )
            )

    def test_existing_checkpoint_requires_explicit_overwrite(self):
        source = self.make_trained_policy()

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            path = Path(temp_dir) / "policy.pt"
            source.save_checkpoint(path, metadata={"revision": 1})

            with self.assertRaises(FileExistsError):
                source.save_checkpoint(path, metadata={"revision": 2})

            source.save_checkpoint(
                path,
                metadata={"revision": 2},
                overwrite=True,
            )
            restored = self.make_policy()
            metadata = restored.load_checkpoint(path)

        self.assertEqual(metadata, {"revision": 2})

    def test_incompatible_dimensions_and_format_are_rejected(self):
        source = self.make_trained_policy()

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            path = Path(temp_dir) / "policy.pt"
            source.save_checkpoint(path)

            for name, restored in (
                ("state_dim", self.make_policy(state_dim=4)),
                ("action_dim", self.make_policy(action_dim=2)),
            ):
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, name):
                        restored.load_checkpoint(path)

            incompatible_path = Path(temp_dir) / "future.pt"
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
            checkpoint["format_version"] = 999
            torch.save(checkpoint, incompatible_path)
            with self.assertRaisesRegex(
                ValueError,
                "format_version.*999",
            ):
                self.make_policy().load_checkpoint(incompatible_path)

    def test_new_and_legacy_loads_emit_no_torch_load_future_warning(self):
        source = self.make_trained_policy()

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            checkpoint_path = Path(temp_dir) / "policy.pt"
            legacy_prefix = str(Path(temp_dir) / "legacy")
            source.save_checkpoint(checkpoint_path)
            source.save(legacy_prefix)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.make_policy().load_checkpoint(checkpoint_path)
                self.make_policy().load(legacy_prefix)

        torch_load_warnings = [
            warning
            for warning in caught
            if issubclass(warning.category, FutureWarning)
            and "torch.load" in str(warning.message)
        ]
        self.assertEqual(torch_load_warnings, [])

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is not available",
    )
    def test_checkpoint_transfers_between_cpu_and_cuda(self):
        source = self.make_trained_policy(device="cpu")
        state = np.array([0.25, -0.5, 0.75], dtype=np.float32)
        expected_action = source.select_action(state)

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            cpu_path = Path(temp_dir) / "cpu.pt"
            gpu_path = Path(temp_dir) / "gpu.pt"
            source.save_checkpoint(cpu_path, metadata={"device": "cpu"})

            gpu_policy = self.make_policy(device="cuda:0")
            metadata = gpu_policy.load_checkpoint(
                cpu_path,
                load_optimizers=True,
            )
            self.assertEqual(metadata, {"device": "cpu"})
            self.assert_networks_on_device(gpu_policy, "cuda:0")
            np.testing.assert_allclose(
                gpu_policy.select_action(state),
                expected_action,
                rtol=1e-5,
                atol=1e-6,
            )

            gpu_policy.save_checkpoint(gpu_path)
            cpu_policy = self.make_policy(device="cpu")
            cpu_policy.load_checkpoint(gpu_path, load_optimizers=True)

        self.assert_networks_equal(gpu_policy, cpu_policy)
        self.assert_networks_on_device(cpu_policy, "cpu")


if __name__ == "__main__":
    unittest.main()
