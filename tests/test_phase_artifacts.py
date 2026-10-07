"""Regression tests for the Task 1 Phase-artifact fixes.

These tests lock in the production behaviour that came out of the Phase 2
incident:

* metrics CSVs are Phase-scoped, so each Phase keeps its own header/schema;
* ``saveLossesToCSV`` refuses to append rows whose keys do not match the
  on-disk header (ordered comparison) instead of silently writing values under
  the wrong column names;
* plots are Phase-scoped and never overwrite another Phase's HTML;
* the Phase 2 shutdown order is ``finalize -> verify frozenPrior -> save final
  checkpoint -> plotMetrics``, and a plotting failure propagates instead of
  blocking (or being hidden by) the critical checkpoint.

Everything is written under ``tmp_path``; no repository metrics/plots/checkpoint
artifact is read, created or modified.
"""

import csv
import hashlib
import os
from pathlib import Path

import pytest
import torch
from pandas.errors import ParserError

import main as main_mod
from dreamer import DreamerV4
from utils import plotMetrics, saveLossesToCSV

# Reuse the project's existing minimal offline run helper instead of
# duplicating a Dreamer config.
from test_offline import _run_config


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

PHASE_COLUMNS = {
    "phase1a": ["envSteps", "gradientSteps", "mse", "lpips", "tokenizerLoss", "psnr"],
    "phase1b": ["envSteps", "gradientSteps", "flowLoss", "flowRegression", "flowBootstrap"],
    "phase2": [
        "envSteps",
        "gradientSteps",
        "flowLoss",
        "flowRegression",
        "flowBootstrap",
        "bcLoss",
        "rewardLoss",
        "phase2Loss",
    ],
    "phase3": ["envSteps", "gradientSteps", "phase3Loss", "pmpoloss", "valueLoss", "advantages"],
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_phase_csv(path, columns, rows=3):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in range(rows):
            writer.writerow(
                [row * 10 if column == "gradientSteps" else float(row + 1) for column in columns]
            )
    return path


def _artifact_config(tmp_path, phase, *, steps=1, save_metrics=True, save_checkpoints=False):
    """Minimal offline run config whose artifact folders all live under tmp."""
    config = _run_config(tmp_path, n=64)
    config.phase = phase
    config.phase1aSteps = steps if phase == "phase1a" else 0
    config.phase1bSteps = steps if phase == "phase1b" else 0
    config.phase2Steps = steps if phase == "phase2" else 0
    config.phase3Steps = steps if phase == "phase3" else 0
    config.checkpointInterval = 1000  # no intermediate checkpoint in 1-step runs
    config.saveCheckpoints = save_checkpoints
    config.saveMetrics = save_metrics
    config.folderNames.metricsFolder = str(tmp_path / "metrics")
    config.folderNames.plotsFolder = str(tmp_path / "plots")
    config.folderNames.checkpointsFolder = str(tmp_path / "ckpts")
    config.folderNames.videosFolder = str(tmp_path / "videos")
    return config


def _assert_nested_equal(first, second, path="checkpoint"):
    """Strict recursive comparison (tensors via ``torch.equal``)."""
    if torch.is_tensor(first):
        assert torch.is_tensor(second), f"{path}: expected a tensor"
        assert first.shape == second.shape, f"{path}: shape {first.shape} != {second.shape}"
        assert first.dtype == second.dtype, f"{path}: dtype {first.dtype} != {second.dtype}"
        assert torch.equal(first, second), f"{path}: tensor values differ"
    elif isinstance(first, dict):
        assert isinstance(second, dict), f"{path}: expected a dict"
        assert set(first) == set(second), (
            f"{path}: keys differ; only-first={set(first) - set(second)}, "
            f"only-second={set(second) - set(first)}"
        )
        for key in first:
            _assert_nested_equal(first[key], second[key], f"{path}.{key}")
    elif isinstance(first, (list, tuple)):
        assert type(first) is type(second), f"{path}: sequence type differs"
        assert len(first) == len(second), f"{path}: length {len(first)} != {len(second)}"
        for index, (left, right) in enumerate(zip(first, second)):
            _assert_nested_equal(left, right, f"{path}[{index}]")
    else:
        assert first == second, f"{path}: {first!r} != {second!r}"


# ---------------------------------------------------------------------------
# 1. Per-Phase metrics CSV isolation
# ---------------------------------------------------------------------------

def test_each_phase_writes_its_own_metrics_csv(monkeypatch, tmp_path):
    """The four Phases must resolve to four distinct, Phase-suffixed CSVs."""
    # Phase 3 normally requires a frozen prior; replace the step so this test
    # only exercises artifact routing, not training math or Phase 3 logic.
    monkeypatch.setattr(
        DreamerV4,
        "trainPhase3",
        lambda self, batch: {
            "phase3Loss": 1.0,
            "pmpoloss": 1.0,
            "valueLoss": 1.0,
            "advantages": 0.0,
        },
    )
    monkeypatch.setattr(main_mod, "plotMetrics", lambda *args, **kwargs: None)

    recorded = []
    real_save = main_mod.saveLossesToCSV

    def spy_save(filename, metrics):
        recorded.append((filename, tuple(metrics.keys())))
        return real_save(filename, metrics)

    monkeypatch.setattr(main_mod, "saveLossesToCSV", spy_save)

    run_name = None
    for phase in PHASE_COLUMNS:
        config = _artifact_config(tmp_path, phase)
        run_name = f"{config.environmentName}_{config.runName}"
        main_mod.run_training(config, "offline")

    filenames = [filename for filename, _ in recorded]
    assert len(filenames) == 4, f"expected one metrics write per Phase, got {filenames}"
    assert len({os.path.abspath(name) for name in filenames}) == 4, (
        f"Phase metrics files are not distinct: {filenames}"
    )
    for phase in PHASE_COLUMNS:
        assert any(phase in os.path.basename(name) for name in filenames), (
            f"no metrics file contains the phase name {phase}: {filenames}"
        )

    # No legacy mixed-schema file (``<run_name>.csv``) may be produced.
    legacy = tmp_path / "metrics" / f"{run_name}.csv"
    assert not legacy.exists(), "a Phase-less mixed metrics CSV was created"

    # Each Phase file exists, its header equals the keys actually written, and
    # every data row has the header's field count (no positional drift).
    for phase in PHASE_COLUMNS:
        phase_name = next(name for name in filenames if phase in os.path.basename(name))
        keys = next(keys for name, keys in recorded if name == phase_name)
        with open(phase_name + ".csv", newline="") as handle:
            rows = list(csv.reader(handle))
        assert rows[0] == list(keys), f"{phase}: header {rows[0]} != keys {list(keys)}"
        assert all(len(row) == len(rows[0]) for row in rows[1:]), (
            f"{phase}: ragged row in {phase_name}.csv"
        )

    # The Phases genuinely have different schemas (so sharing a file would mix
    # headers); at least the three trained Phases differ from one another.
    headers = {}
    for phase in PHASE_COLUMNS:
        phase_name = next(name for name in filenames if phase in os.path.basename(name))
        headers[phase] = next(keys for name, keys in recorded if name == phase_name)
    assert headers["phase1a"] != headers["phase1b"]
    assert headers["phase1b"] != headers["phase2"]
    assert headers["phase1a"] != headers["phase2"]


# ---------------------------------------------------------------------------
# 2. CSV header / metrics-key consistency (ordered, complete comparison)
# ---------------------------------------------------------------------------

def test_first_write_header_equals_metrics_keys(tmp_path):
    base = str(tmp_path / "phaseA")
    saveLossesToCSV(base, {"envSteps": 0, "gradientSteps": 1, "mse": 0.5})
    with open(base + ".csv", newline="") as handle:
        header = next(csv.reader(handle))
    assert header == ["envSteps", "gradientSteps", "mse"]


def test_same_schema_append_keeps_header_order(tmp_path):
    base = str(tmp_path / "phaseB")
    saveLossesToCSV(base, {"envSteps": 0, "gradientSteps": 1, "bcLoss": 0.1, "rewardLoss": 0.2})
    saveLossesToCSV(base, {"envSteps": 0, "gradientSteps": 2, "bcLoss": 0.3, "rewardLoss": 0.4})
    with open(base + ".csv", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["envSteps", "gradientSteps", "bcLoss", "rewardLoss"]
    assert rows[1] == ["0", "1", "0.1", "0.2"]
    assert rows[2] == ["0", "2", "0.3", "0.4"]


@pytest.mark.parametrize(
    "mismatched",
    [
        {"envSteps": 0, "gradientSteps": 2, "rewardLoss": 0.4, "bcLoss": 0.3},  # reordered
        {"envSteps": 0, "gradientSteps": 2, "bcLoss": 0.3},  # key missing
        {
            "envSteps": 0,
            "gradientSteps": 2,
            "bcLoss": 0.3,
            "rewardLoss": 0.4,
            "phase2Loss": 0.5,
        },  # key added
        {"envSteps": 0, "gradientSteps": 2, "bcLoss": 0.3, "flowLoss": 0.0},  # key renamed
    ],
)
def test_schema_mismatch_refuses_append_and_preserves_file(tmp_path, mismatched):
    base = str(tmp_path / "phaseC")
    saveLossesToCSV(base, {"envSteps": 0, "gradientSteps": 1, "bcLoss": 0.1, "rewardLoss": 0.2})
    before = Path(base + ".csv").read_bytes()

    with pytest.raises(ValueError) as excinfo:
        saveLossesToCSV(base, mismatched)

    message = str(excinfo.value)
    # The message must expose both the on-disk and incoming schemas.
    assert "bcLoss" in message and "rewardLoss" in message, message
    assert "header" in message, message
    assert Path(base + ".csv").read_bytes() == before, "CSV changed after a refused write"


# ---------------------------------------------------------------------------
# 3. Every Phase CSV is parseable by plotMetrics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phase", list(PHASE_COLUMNS))
def test_phase_csv_is_plottable(tmp_path, phase):
    csv_base = tmp_path / "metrics" / f"FakeEnv-v0_test_{phase}"
    _write_phase_csv(str(csv_base) + ".csv", PHASE_COLUMNS[phase])
    html_base = tmp_path / "plots" / f"FakeEnv-v0_test_{phase}"
    html_base.parent.mkdir(parents=True, exist_ok=True)

    plotMetrics(f"{csv_base}", title=phase, savePath=f"{html_base}")

    html = Path(str(html_base) + ".html")
    assert html.exists(), f"{phase}: no HTML was written"
    assert html.stat().st_size > 0, f"{phase}: HTML is empty"
    assert phase in html.name, f"{phase}: HTML path {html.name} lacks the phase name"


def test_plot_metrics_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        plotMetrics(str(tmp_path / "does_not_exist"), savePath=str(tmp_path / "plot"))


def test_plot_metrics_malformed_schema_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    # Header has 3 fields, second data row has 5 -> unparseable.
    bad.write_text("envSteps,gradientSteps,mse\n0,1,1\n0,2,1,2,3\n")
    with pytest.raises(ParserError):
        plotMetrics(str(bad), savePath=str(tmp_path / "plot"))


# ---------------------------------------------------------------------------
# 4. Phase plots do not overwrite each other
# ---------------------------------------------------------------------------

def test_phase_plots_do_not_overwrite_each_other(tmp_path):
    paths = {}
    for phase, columns in PHASE_COLUMNS.items():
        csv_path = _write_phase_csv(tmp_path / f"{phase}.csv", columns)
        html_base = tmp_path / "plots" / f"FakeEnv-v0_test_{phase}"
        html_base.parent.mkdir(parents=True, exist_ok=True)
        plotMetrics(str(csv_path), title=phase, savePath=str(html_base))
        paths[phase] = Path(str(html_base) + ".html")

    assert len({str(path) for path in paths.values()}) == 4
    for phase, path in paths.items():
        assert path.exists(), f"{phase}: HTML missing"
        assert phase in path.name, f"{phase}: HTML path lacks the phase name"

    hashes = {phase: _sha256(path) for phase, path in paths.items()}

    # Regenerate only phase1a; the other three files must be untouched.
    plotMetrics(
        str(tmp_path / "phase1a.csv"),
        title="phase1a",
        savePath=str(tmp_path / "plots" / "FakeEnv-v0_test_phase1a"),
    )
    for phase in ("phase1b", "phase2", "phase3"):
        assert _sha256(paths[phase]) == hashes[phase], (
            f"regenerating phase1a changed the {phase} plot"
        )


# ---------------------------------------------------------------------------
# 5. Plot failure happens after the final checkpoint is persisted
# ---------------------------------------------------------------------------

def test_final_checkpoint_saved_before_plot_failure(monkeypatch, tmp_path):
    config = _artifact_config(tmp_path, "phase2", save_checkpoints=True)
    events = []

    real_finalize = DreamerV4.finalizePhase2
    real_save_checkpoint = DreamerV4.saveCheckpoint
    real_save_final = DreamerV4.saveFinalCheckpoint

    def spy_finalize(self):
        events.append("finalize")
        result = real_finalize(self)
        events.append("prior_verified" if self.frozenPrior is not None else "prior_missing")
        return result

    save_calls = []

    def spy_save(self, path):
        save_calls.append(path)
        return real_save_checkpoint(self, path)

    def spy_save_final(self, path):
        events.append("save_final")
        return real_save_final(self, path)

    def boom(*args, **kwargs):
        events.append("plot")
        raise RuntimeError("plotMetrics boom")

    monkeypatch.setattr(DreamerV4, "finalizePhase2", spy_finalize)
    monkeypatch.setattr(DreamerV4, "saveCheckpoint", spy_save)
    monkeypatch.setattr(DreamerV4, "saveFinalCheckpoint", spy_save_final)
    monkeypatch.setattr(main_mod, "plotMetrics", boom)

    with pytest.raises(RuntimeError, match="plotMetrics boom"):
        main_mod.run_training(config, "offline")

    # The final checkpoint must go through the no-overwrite entry point, not the
    # regular (overwrite-allowed) one.  Recording distinct event labels makes a
    # regression back to saveCheckpoint observable (Task 3 review MEDIUM-2).
    assert events == ["finalize", "prior_verified", "save_final", "plot"], events
    assert save_calls == [], (
        "run_training used saveCheckpoint (overwrite-allowed) for the final "
        f"Phase 2 checkpoint: {save_calls}"
    )

    run_name = f"{config.environmentName}_{config.runName}"
    finals = list((tmp_path / "ckpts").glob(f"{run_name}_*_final.pth"))
    assert len(finals) == 1, f"expected exactly one final checkpoint, got {finals}"

    # The persisted final checkpoint must be loadable through the production
    # loader and must carry a frozen prior.
    reloaded = DreamerV4(
        (3, 8, 8),
        2,
        config.actionLow,
        config.actionHigh,
        torch.device("cpu"),
        config.dreamer,
    )
    reloaded.loadCheckpoint(str(finals[0]))
    assert reloaded.frozenPrior is not None, "final checkpoint has no frozen prior"
    assert not any(p.requires_grad for p in reloaded.frozenPrior.parameters())


# ---------------------------------------------------------------------------
# 6-7. finalizePhase2 does not train: no optimizer.step, step counter unchanged
# ---------------------------------------------------------------------------

def test_finalize_does_not_call_optimizer_step(monkeypatch, small_dreamer, small_batch):
    small_dreamer.trainPhase2(small_batch)
    calls = []

    def fail_step(self, *args, **kwargs):
        calls.append(self)
        raise AssertionError("finalizePhase2 must not call optimizer.step")

    monkeypatch.setattr(torch.optim.Adam, "step", fail_step)

    small_dreamer.finalizePhase2()

    assert calls == [], "finalizePhase2 triggered an optimizer step"
    assert small_dreamer.frozenPrior is not None


def test_finalize_preserves_total_gradient_steps(small_dreamer, small_batch):
    small_dreamer.trainPhase2(small_batch)
    small_dreamer.total_gradient_steps = 320000

    small_dreamer.finalizePhase2()

    assert small_dreamer.total_gradient_steps == 320000


# ---------------------------------------------------------------------------
# 8. Only frozenPrior changes across finalize
# ---------------------------------------------------------------------------

def test_finalize_changes_only_frozen_prior(small_dreamer, small_batch, tmp_path):
    small_dreamer.trainPhase2(small_batch)

    before_path = tmp_path / "before_finalize"
    small_dreamer.saveCheckpoint(str(before_path))
    before = torch.load(str(before_path) + ".pth", map_location="cpu")

    small_dreamer.finalizePhase2()

    after_path = tmp_path / "after_finalize"
    small_dreamer.saveCheckpoint(str(after_path))
    after = torch.load(str(after_path) + ".pth", map_location="cpu")

    assert set(before) == set(after)
    assert before["frozenPrior"] is None
    assert after["frozenPrior"] is not None

    changed = []
    for key in before:
        if key == "frozenPrior":
            continue
        try:
            _assert_nested_equal(before[key], after[key], key)
        except AssertionError:  # pragma: no cover - failure reporting aid
            changed.append(key)
    assert changed == [], f"finalize changed non-frozenPrior fields: {changed}"


# ---------------------------------------------------------------------------
# 9-10. frozenPrior equals policyHead but shares no storage
# ---------------------------------------------------------------------------

def test_frozen_prior_matches_policy_head_state(small_dreamer, small_batch):
    small_dreamer.trainPhase2(small_batch)
    small_dreamer.finalizePhase2()

    prior = small_dreamer.frozenPrior
    assert prior is not None
    prior_state = prior.state_dict()
    policy_state = small_dreamer.policyHead.state_dict()

    assert set(prior_state) == set(policy_state)
    for key in policy_state:
        assert prior_state[key].shape == policy_state[key].shape, key
        assert prior_state[key].dtype == policy_state[key].dtype, key
        assert prior_state[key].device == policy_state[key].device, key
        assert torch.equal(prior_state[key], policy_state[key]), key

    assert all(not parameter.requires_grad for parameter in prior.parameters())
    assert prior.training is False


def test_frozen_prior_does_not_share_storage(small_dreamer, small_batch):
    small_dreamer.trainPhase2(small_batch)
    small_dreamer.finalizePhase2()

    prior_parameters = dict(small_dreamer.frozenPrior.named_parameters())
    policy_parameters = dict(small_dreamer.policyHead.named_parameters())

    assert set(prior_parameters) == set(policy_parameters)
    assert len(prior_parameters) > 0

    for name, prior_parameter in prior_parameters.items():
        policy_parameter = policy_parameters[name]
        assert prior_parameter.data_ptr() != policy_parameter.data_ptr(), name
        assert (
            prior_parameter.untyped_storage().data_ptr()
            != policy_parameter.untyped_storage().data_ptr()
        ), name


# ---------------------------------------------------------------------------
# 11. final checkpoint round-trips through the production loadCheckpoint
# ---------------------------------------------------------------------------

def test_final_checkpoint_roundtrips_via_load_checkpoint(small_config, small_batch, tmp_path):
    source = DreamerV4(
        (3, 64, 64),
        3,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        torch.device("cpu"),
        small_config,
    )
    source.trainPhase2(small_batch)
    source.finalizePhase2()
    source.total_gradient_steps = 320000

    final_path = tmp_path / "final"
    source.saveCheckpoint(str(final_path))

    target = DreamerV4(
        (3, 64, 64),
        3,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        torch.device("cpu"),
        small_config,
    )
    target.loadCheckpoint(str(final_path))

    assert target.total_gradient_steps == 320000
    assert target.frozenPrior is not None
    for name in ("tokenizer", "dynamics", "policyHead", "rewardHead", "valueHead", "frozenPrior"):
        source_state = getattr(source, name).state_dict()
        target_state = getattr(target, name).state_dict()
        for key, value in source_state.items():
            assert torch.equal(value, target_state[key]), f"{name}.{key}"


# ---------------------------------------------------------------------------
# Plot wiring: run_training must pass the Phase-scoped CSV/HTML paths
# ---------------------------------------------------------------------------

def test_run_training_passes_phase_scoped_plot_paths(monkeypatch, tmp_path):
    """run_training must hand plotMetrics the matching Phase CSV and HTML path.

    This replaces a spy that discarded ``*args, **kwargs`` and therefore could
    not catch a wiring regression (GLM-5.3 review, MEDIUM-1).
    """
    monkeypatch.setattr(
        DreamerV4,
        "trainPhase3",
        lambda self, batch: {
            "phase3Loss": 1.0,
            "pmpoloss": 1.0,
            "valueLoss": 1.0,
            "advantages": 0.0,
        },
    )

    captured = []

    def spy_plot(filename, title="", savePath="metricsPlot", window=10):
        captured.append({"filename": filename, "title": title, "savePath": savePath})

    monkeypatch.setattr(main_mod, "plotMetrics", spy_plot)

    for phase in PHASE_COLUMNS:
        config = _artifact_config(tmp_path, phase)
        main_mod.run_training(config, "offline")

    assert len(captured) == 4, captured
    run_name = "FakeEnv-v0_test"
    for phase, call in zip(PHASE_COLUMNS, captured):
        expected_csv = str(tmp_path / "metrics" / f"{run_name}_{phase}")
        expected_html = str(tmp_path / "plots" / f"{run_name}_{phase}")
        legacy_csv = str(tmp_path / "metrics" / run_name)
        legacy_html = str(tmp_path / "plots" / run_name)

        assert call["filename"] == expected_csv, (phase, call)
        assert call["savePath"] == expected_html, (phase, call)
        assert call["title"] == f"FakeEnv-v0_{phase}", (phase, call)
        assert phase in call["filename"] and phase in call["savePath"]
        assert call["filename"] != legacy_csv, "plotMetrics read the legacy mixed CSV"
        assert call["savePath"] != legacy_html, "plotMetrics wrote the legacy HTML path"


# ---------------------------------------------------------------------------
# Task 4: no-replace publish is real behaviour (replaces the old XFAIL)
# ---------------------------------------------------------------------------

def test_recovery_publish_refuses_to_overwrite_existing_target(tmp_path):
    """The publish helper must refuse an existing target and leave it intact.

    This replaces the Task 3 XFAIL that only probed ``publishFinalCheckpoint``
    existence: it now exercises the production helper (``utils.publishFileNoReplace``)
    and asserts the strict no-overwrite contract at the byte/hash level.
    """
    from utils import publishFileNoReplace

    candidate = tmp_path / "candidate.tmp"
    candidate.write_bytes(b"candidate-payload")
    target = tmp_path / "run_320k_final.pth"
    target.write_bytes(b"original-final-bytes")
    before_hash = _sha256(target)

    with pytest.raises(FileExistsError):
        publishFileNoReplace(str(candidate), str(target))

    assert target.read_bytes() == b"original-final-bytes", "refused publish changed the target"
    assert _sha256(target) == before_hash, "refused publish changed the target hash"
    assert not candidate.exists(), "refused publish left the candidate temp behind"
    assert list(tmp_path.glob("*.tmp")) == [], "refused publish left a temp file"


def test_recovery_publish_creates_target_atomically(tmp_path):
    """A successful no-replace publish moves the candidate onto the target."""
    from utils import publishFileNoReplace

    candidate = tmp_path / "candidate.tmp"
    candidate.write_bytes(b"candidate-payload")
    target = tmp_path / "run_320k_final.pth"

    publishFileNoReplace(str(candidate), str(target))

    assert target.read_bytes() == b"candidate-payload"
    assert not candidate.exists(), "published candidate name was not cleaned up"
    assert list(tmp_path.glob("*.tmp")) == []
