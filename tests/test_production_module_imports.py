import importlib
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_PACKAGES = (
    "agents",
    "environments",
    "evaluation",
    "experiments",
    "problems",
    "swarm",
    "training",
)


def discover_production_modules():
    modules = set()
    for package_name in SUPPORTED_PACKAGES:
        modules.add(package_name)
        package_path = PROJECT_ROOT / package_name
        for source_path in package_path.rglob("*.py"):
            if "__pycache__" in source_path.parts:
                continue
            relative_path = source_path.relative_to(PROJECT_ROOT)
            if relative_path.name == "__init__.py":
                module_parts = relative_path.parts[:-1]
            else:
                module_parts = relative_path.with_suffix("").parts
            modules.add(".".join(module_parts))
    return sorted(modules)


class TestProductionModuleImports(unittest.TestCase):
    def test_all_supported_production_modules_import(self):
        modules = discover_production_modules()
        self.assertTrue(modules)
        self.assertTrue(set(SUPPORTED_PACKAGES).issubset(modules))

        original_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            for module_name in modules:
                with self.subTest(module=module_name):
                    imported = importlib.import_module(module_name)
                    self.assertEqual(imported.__name__, module_name)
        finally:
            sys.dont_write_bytecode = original_dont_write_bytecode


if __name__ == "__main__":
    unittest.main()
