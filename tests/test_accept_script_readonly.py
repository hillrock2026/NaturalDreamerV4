"""Task 6 regression tests: the acceptance script must be strictly read-only.

These tests lock in the Task 6 fix for
``tools/accept_320k_final.py``: the permission policy is *checked* and never
*repaired*, and the script contains no forbidden mutating call.  Nothing here
touches the real checkpoints; all filesystem work uses ``tmp_path``.
"""

import ast
import importlib.util
import os
import stat
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "accept_320k_final.py"

# Attributes that must never be *called* by the read-only acceptance script.
FORBIDDEN_ATTR_CALLS = {
    "chmod", "replace", "rename", "renames",
    "copy", "copyfile", "copy2", "copytree",
    "rmtree", "remove", "unlink",
    "saveCheckpoint", "saveFinalCheckpoint", "finalizePhase2",
    "trainPhase1a", "trainPhase1b", "trainPhase2", "trainPhase3",
    "environmentInteraction", "run_training", "step",
}
FORBIDDEN_NAME_CALLS = {
    "saveCheckpoint", "saveFinalCheckpoint", "finalizePhase2",
    "trainPhase1a", "trainPhase1b", "trainPhase2", "trainPhase3",
    "environmentInteraction", "run_training",
}


@pytest.fixture(scope="module")
def accept_module():
    spec = importlib.util.spec_from_file_location("accept_320k_final_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _forbidden_calls(tree):
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_ATTR_CALLS:
            found.append(f"line {node.lineno}: .{func.attr}()")
        if isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_NAME_CALLS:
            found.append(f"line {node.lineno}: .{func.attr}()")
        if isinstance(func, ast.Name) and func.id in FORBIDDEN_NAME_CALLS:
            found.append(f"line {node.lineno}: {func.id}()")
    return found


def test_accept_script_has_no_forbidden_calls():
    """Static AST scan: no chmod/replace/copy/train/save/finalize/step calls."""
    tree = ast.parse(SCRIPT.read_text())
    found = _forbidden_calls(tree)
    assert found == [], f"forbidden mutating calls in {SCRIPT.name}: {found}"


def test_accept_script_contains_no_chmod_token_outside_guards():
    """The script must not even reference ``os.chmod`` as an attribute access."""
    tree = ast.parse(SCRIPT.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "chmod":
            raise AssertionError(f"'os.chmod' attribute access at line {node.lineno}")


def test_permission_policy_accepts_aligned_modes(accept_module):
    for mode in ("0o644", "0o664"):
        verdict = accept_module.permissionPolicyCheck("0o664", mode)
        assert verdict["ok"] is True, verdict


def test_permission_policy_rejects_owner_only_mode(accept_module):
    verdict = accept_module.permissionPolicyCheck("0o664", "0o600")
    assert verdict["ok"] is False
    assert "MISMATCH" in verdict["decision"]


def test_permission_check_never_mutates_mode(accept_module, tmp_path):
    """A mismatch must be reported, never repaired (no chmod, no metadata change)."""
    target = tmp_path / "final.pth"
    target.write_bytes(b"content")
    os.chmod(target, 0o600)
    before_mode = stat.S_IMODE(os.stat(target).st_mode)
    after_bytes = target.read_bytes()

    verdict = accept_module.permissionPolicyCheck("0o664", oct(before_mode))

    assert verdict["ok"] is False
    assert stat.S_IMODE(os.stat(target).st_mode) == before_mode, "check changed the file mode"
    assert target.read_bytes() == after_bytes, "check changed the file content"


def test_guards_fail_closed_on_forbidden_call(accept_module):
    guards = accept_module.Guards()
    with pytest.raises(AssertionError):
        guards._fail("unit-test")
    assert guards.violations == ["unit-test"]


def test_optimizer_step_monitor_counts_and_raises(accept_module):
    class DummyOptimizer:
        def step(self, *args, **kwargs):
            return "stepped"

    guards = accept_module.Guards()
    optimizer = DummyOptimizer()
    monitor = accept_module.OptimizerStepMonitor("dummy", optimizer, guards)

    with pytest.raises(AssertionError):
        optimizer.step()

    assert monitor.count == 1
    assert guards.optimizer_steps == 1
    assert guards.violations


def test_install_guards_makes_training_and_save_calls_raise(accept_module):
    import torch

    import main as main_mod
    from dreamer import DreamerV4

    method_names = (
        "trainPhase1a", "trainPhase1b", "trainPhase2", "trainPhase3",
        "environmentInteraction", "finalizePhase2",
        "saveCheckpoint", "saveFinalCheckpoint",
    )
    originals = {name: getattr(DreamerV4, name) for name in method_names}
    original_adam_step = torch.optim.Adam.step
    original_run_training = main_mod.run_training

    guards = accept_module.Guards()
    try:
        accept_module.installGuards(guards)
        with pytest.raises(AssertionError):
            DreamerV4.trainPhase3(object(), None)
        with pytest.raises(AssertionError):
            DreamerV4.saveFinalCheckpoint(object(), "x")
        with pytest.raises(AssertionError):
            DreamerV4.saveCheckpoint(object(), "x")
        with pytest.raises(AssertionError):
            DreamerV4.finalizePhase2(object())
        assert guards.phase_calls["phase3"] == 1
        assert guards.forbidden["saveFinalCheckpoint"] == 1
        assert guards.forbidden["saveCheckpoint"] == 1
        assert guards.forbidden["finalizePhase2"] == 1
    finally:
        # Restore in place (no module reload) so other tests keep the real class.
        for name, original in originals.items():
            setattr(DreamerV4, name, original)
        torch.optim.Adam.step = original_adam_step
        main_mod.run_training = original_run_training
