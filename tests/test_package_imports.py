import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SimulationPackageImportTest(unittest.TestCase):
    def _run_blocked_import(self, module_name: str):
        code = f"""
import builtins
real_import = builtins.__import__

def blocked(name, globals=None, locals=None, fromlist=(), level=0):
    if name.startswith("langchain") or name.startswith("langchain_community"):
        raise ImportError("blocked for import-boundary test")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked
import {module_name}
print("ok")
"""
        return subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    def test_importing_simulation_control_core_does_not_pull_rag_dependencies(self):
        proc = self._run_blocked_import("simulation.control_core")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ok", proc.stdout)

    def test_importing_simulation_env_does_not_pull_rag_dependencies(self):
        proc = self._run_blocked_import("simulation.env")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ok", proc.stdout)


if __name__ == "__main__":
    unittest.main()
