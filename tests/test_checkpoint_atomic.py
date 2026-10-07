"""Atomic checkpoint persistence regression tests (Task 3).

Covers the write -> flush -> fsync -> ``os.replace`` -> directory-fsync
pipeline used by ``DreamerV4.saveCheckpoint`` (overwrite allowed) and
``DreamerV4.saveFinalCheckpoint`` (no-overwrite).  Every test uses ``tmp_path``;
no repository checkpoint is touched.
"""

import os
from pathlib import Path

import pytest
import torch

import utils
from dreamer import DreamerV4


def _clone(small_dreamer):
    return DreamerV4(
        small_dreamer.observation_shape,
        small_dreamer.action_size,
        small_dreamer.action_low,
        small_dreamer.action_high,
        torch.device("cpu"),
        small_dreamer.config,
    )


def _assert_state_dicts_exact(first, second):
    first_state = first.state_dict()
    second_state = second.state_dict()
    assert set(first_state) == set(second_state)
    for key in first_state:
        assert torch.equal(first_state[key], second_state[key]), key


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------

def test_atomic_save_success_loads_and_leaves_no_temp(small_dreamer, small_batch, tmp_path):
    small_dreamer.trainPhase2(small_batch)
    target = tmp_path / "ckpts" / "run_1steps"

    small_dreamer.saveCheckpoint(str(target))

    final = Path(str(target) + ".pth")
    assert final.exists()
    assert final.stat().st_size > 0
    assert list((tmp_path / "ckpts").glob("*.tmp")) == [], "temp file was left behind"

    reloaded = _clone(small_dreamer)
    reloaded.loadCheckpoint(str(target))
    assert reloaded.total_gradient_steps == small_dreamer.total_gradient_steps
    for name in ("tokenizer", "dynamics", "policyHead", "rewardHead", "valueHead"):
        _assert_state_dicts_exact(getattr(small_dreamer, name), getattr(reloaded, name))


def test_save_final_checkpoint_success_and_no_overwrite(small_dreamer, tmp_path):
    target = tmp_path / "run_320k_final"

    small_dreamer.saveFinalCheckpoint(str(target))

    final = Path(str(target) + ".pth")
    assert final.exists()

    reloaded = _clone(small_dreamer)
    reloaded.loadCheckpoint(str(target))
    assert reloaded.total_gradient_steps == small_dreamer.total_gradient_steps

    # Re-publishing over an existing final must fail before any write.
    before = final.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite=False"):
        small_dreamer.saveFinalCheckpoint(str(target))
    assert final.read_bytes() == before, "refused publish modified the existing target"
    assert list(tmp_path.glob("*.tmp")) == [], "refused publish created a temp file"
    assert len(list(tmp_path.glob("*_final.pth"))) == 1


def test_save_checkpoint_keeps_overwrite_semantics(small_dreamer, tmp_path):
    """Regular checkpoints must remain re-writable (legacy behaviour)."""
    target = tmp_path / "run_1steps"
    small_dreamer.saveCheckpoint(str(target))

    small_dreamer.total_gradient_steps = 123
    small_dreamer.saveCheckpoint(str(target))

    saved = torch.load(str(target) + ".pth", map_location="cpu")
    assert saved["totalGradientSteps"] == 123
    assert list(tmp_path.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# Write failure
# ---------------------------------------------------------------------------

def test_write_failure_leaves_no_target_and_cleans_temp(monkeypatch, small_dreamer, tmp_path):
    def boom(*args, **kwargs):
        raise RuntimeError("torch.save boom")

    monkeypatch.setattr(torch, "save", boom)
    target = tmp_path / "run_1steps"

    with pytest.raises(RuntimeError, match="torch.save boom"):
        small_dreamer.saveCheckpoint(str(target))

    assert not (tmp_path / "run_1steps.pth").exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_failure_preserves_existing_target(monkeypatch, small_dreamer, tmp_path):
    target = tmp_path / "run_1steps"
    small_dreamer.saveCheckpoint(str(target))
    final = Path(str(target) + ".pth")
    before = final.read_bytes()

    def boom(*args, **kwargs):
        raise RuntimeError("torch.save boom")

    monkeypatch.setattr(torch, "save", boom)

    with pytest.raises(RuntimeError, match="torch.save boom"):
        small_dreamer.saveCheckpoint(str(target))

    assert final.read_bytes() == before, "failed overwrite truncated the existing target"
    assert list(tmp_path.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# File fsync failure (must happen before publish)
# ---------------------------------------------------------------------------

def test_file_fsync_failure_does_not_publish(monkeypatch, small_dreamer, tmp_path):
    def failing_fsync(fd):
        raise OSError("fsync boom")

    monkeypatch.setattr(os, "fsync", failing_fsync)
    target = tmp_path / "run_1steps"

    with pytest.raises(OSError, match="fsync boom"):
        small_dreamer.saveCheckpoint(str(target))

    assert not (tmp_path / "run_1steps.pth").exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_file_fsync_failure_preserves_existing_target(monkeypatch, small_dreamer, tmp_path):
    target = tmp_path / "run_1steps"
    small_dreamer.saveCheckpoint(str(target))
    final = Path(str(target) + ".pth")
    before = final.read_bytes()

    def failing_fsync(fd):
        raise OSError("fsync boom")

    monkeypatch.setattr(os, "fsync", failing_fsync)

    with pytest.raises(OSError, match="fsync boom"):
        small_dreamer.saveCheckpoint(str(target))

    assert final.read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# Directory fsync failure (post-publish durability step)
# ---------------------------------------------------------------------------

def test_directory_fsync_failure_propagates_but_target_stays_valid(
    monkeypatch, small_dreamer, tmp_path
):
    """A directory-fsync failure is reported, but the published file remains.

    ``os.replace`` has already succeeded at this point, so the target is a
    complete, valid checkpoint; deleting it to "undo" the publish would be
    destructive and race with readers.  The error is not swallowed.
    """
    def failing_directory_fsync(directory):
        raise OSError("dir fsync boom")

    monkeypatch.setattr(utils, "fsyncDirectory", failing_directory_fsync)
    target = tmp_path / "run_1steps"

    with pytest.raises(OSError, match="dir fsync boom"):
        small_dreamer.saveCheckpoint(str(target))

    final = Path(str(target) + ".pth")
    assert final.exists(), "published file was removed by the durability failure"
    loaded = torch.load(str(final), map_location="cpu")
    assert "tokenizer" in loaded and "dynamics" in loaded
    assert list(tmp_path.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# Atomic replace failure
# ---------------------------------------------------------------------------

def test_replace_failure_cleans_temp_and_leaves_no_new_target(
    monkeypatch, small_dreamer, tmp_path
):
    def failing_replace(src, dst):
        raise OSError("replace boom")

    monkeypatch.setattr(os, "replace", failing_replace)
    target = tmp_path / "run_1steps"

    with pytest.raises(OSError, match="replace boom"):
        small_dreamer.saveCheckpoint(str(target))

    assert not (tmp_path / "run_1steps.pth").exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_replace_failure_preserves_existing_target(monkeypatch, small_dreamer, tmp_path):
    target = tmp_path / "run_1steps"
    small_dreamer.saveCheckpoint(str(target))
    final = Path(str(target) + ".pth")
    before = final.read_bytes()

    def failing_replace(src, dst):
        raise OSError("replace boom")

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError, match="replace boom"):
        small_dreamer.saveCheckpoint(str(target))

    assert final.read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# Helper-level guarantees
# ---------------------------------------------------------------------------

def test_atomic_write_no_overwrite_creates_no_temp(tmp_path):
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"original")

    with pytest.raises(FileExistsError):
        utils.atomicWriteFile(str(target), lambda handle: handle.write(b"new"), overwrite=False)

    assert target.read_bytes() == b"original"
    assert list(tmp_path.glob("*.tmp")) == []


def test_fsync_directory_is_noop_on_windows(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(os, "fsync", lambda fd: calls.append(fd))

    utils.fsyncDirectory(str(tmp_path))

    assert calls == []


# ---------------------------------------------------------------------------
# Atomic no-replace (Task 3 review MEDIUM-1: closes the exists()->replace()
# TOCTOU window using os.link)
# ---------------------------------------------------------------------------

def test_atomic_write_no_overwrite_is_atomic_against_racing_creator(tmp_path, monkeypatch):
    """Simulate the TOCTOU window: the early exists() check is bypassed.

    A concurrent writer's target must survive and ``FileExistsError`` must
    still be raised by the atomic ``os.link`` publish step.
    """
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"racing-writer")
    monkeypatch.setattr(utils.os.path, "exists", lambda path: False)

    with pytest.raises(FileExistsError):
        utils.atomicWriteFile(str(target), lambda handle: handle.write(b"ours"), overwrite=False)

    assert target.read_bytes() == b"racing-writer", "atomic no-replace clobbered the target"
    assert list(tmp_path.glob("*.tmp")) == []


def test_publish_file_no_replace_refuses_existing_target(tmp_path):
    candidate = tmp_path / "candidate.tmp"
    candidate.write_bytes(b"candidate")
    target = tmp_path / "final.pth"
    target.write_bytes(b"original")
    before = target.read_bytes()

    with pytest.raises(FileExistsError):
        utils.publishFileNoReplace(str(candidate), str(target))

    assert target.read_bytes() == before
    assert not candidate.exists(), "refused publish left the candidate behind"
    assert list(tmp_path.glob("*.tmp")) == []


def test_publish_file_no_replace_success_moves_candidate(tmp_path):
    candidate = tmp_path / "candidate.tmp"
    candidate.write_bytes(b"candidate")
    target = tmp_path / "final.pth"

    utils.publishFileNoReplace(str(candidate), str(target))

    assert target.read_bytes() == b"candidate"
    assert not candidate.exists()
    assert list(tmp_path.glob("*.tmp")) == []
