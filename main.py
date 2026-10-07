import argparse
import os

import torch

from buffer import ReplayBuffer
from dreamer import DreamerV4, resolvePrecision
from utils import ensureParentFolders, loadConfig, plotMetrics, saveLossesToCSV, seedEverything

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VALID_MODES = ("online", "offline")


def resolve_mode(config, mode_override=None):
    """Resolve the data-source mode from config and an optional CLI override."""
    mode = mode_override or getattr(config, "mode", "online")
    if mode not in VALID_MODES:
        raise ValueError(
            f"Unknown mode '{mode}'. Must be one of {VALID_MODES}. "
            "Choose 'offline' to train from a .npz dataset without environment interaction."
        )
    return mode


def evaluation_enabled(config, mode):
    """Whether checkpoint-time environment evaluation should run.

    Online mode preserves the original behaviour (always evaluate).  Offline
    mode only evaluates when explicitly requested, so a machine without
    Gymnasium/CarRacing can still train.
    """
    if mode == "online":
        return True
    return bool(getattr(config, "evaluateCheckpoints", False))


def build_pixels_environment(environment_name, render_mode=None):
    """Build the pixel-processing environment, importing gymnasium lazily.

    Gymnasium is imported inside this function (not at module import time) so
    offline training can run on a machine without Gymnasium installed.  A
    missing environment yields a clear error instead of a silent skip.
    """
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise RuntimeError(
            "Gymnasium is not installed, but an environment is required for "
            f"'{environment_name}'. For offline training without evaluation, "
            "set evaluateCheckpoints=false."
        ) from exc
    from envs import CleanGymWrapper, GymPixelsProcessingWrapper

    try:
        env = gym.make(environment_name, render_mode=render_mode)
    except gym.error.Error as exc:
        raise RuntimeError(
            f"Environment '{environment_name}' is not available for evaluation. "
            "For offline training without evaluation, set evaluateCheckpoints=false."
        ) from exc
    return CleanGymWrapper(
        GymPixelsProcessingWrapper(
            gym.wrappers.ResizeObservation(env, (64, 64))
        )
    )


def get_env_properties(env):
    from envs import getEnvProperties

    return getEnvProperties(env)


def load_offline_dataset(dreamer, offline_path, batch_size, max_sequence_size, missing_relevant_policy="uniform"):
    """Load an offline .npz dataset into the dreamer's buffer with validation.

    Raises ``FileNotFoundError`` if the file is missing and ``ValueError`` for
    insufficient data / unsupported sequence lengths.
    """
    if not offline_path:
        raise ValueError(
            "mode='offline' requires a non-empty offlineDataPath pointing to a .npz file."
        )
    n = dreamer.buffer.loadOffline(offline_path, missing_relevant_policy=missing_relevant_policy)
    dreamer.buffer.validateForSampling(batch_size, max_sequence_size)
    print(f"Loaded {n} offline transitions from {offline_path}.")
    return n


def checkpointStepSuffix(total_gradient_steps):
    """Return a checkpoint suffix that is unique for small step counts.

    Steps below 1000 use an exact ``<N>steps`` suffix so that distinct small
    steps cannot collide on a rounded ``0k``.  From 1000 on the historical
    ``<N>k`` convention is preserved for compatibility.
    """
    if total_gradient_steps < 1000:
        return f"{total_gradient_steps}steps"
    return f"{total_gradient_steps/1000:.0f}k"


def checkpointPath(config):
    """Resolve the checkpoint path to load from the run/config naming rules."""
    run_name = f"{config.environmentName}_{config.runName}"
    return os.path.join(
        config.folderNames.checkpointsFolder, f"{run_name}_{config.checkpointToLoad}"
    )


def run_training(config, mode):
    """Run the configured training phase for the given data-source mode.

    ``mode == 'offline'`` never builds a training environment and never calls
    ``environmentInteraction`` for warmup; checkpoint evaluation is optional
    and its environment is created lazily.
    """
    if not config.dreamer.batchLengthLong > config.dreamer.contextLength:
        raise ValueError(
            "T2 (batchLengthLong) must be greater than contextLength C, got "
            f"batchLengthLong={config.dreamer.batchLengthLong} and "
            f"contextLength={config.dreamer.contextLength}."
        )
    # Fail fast on an unsupported precision before building env/model.
    resolvePrecision(getattr(config.dreamer, "precision", "fp32"))

    phase = config.phase
    steps_by_phase = {
        "phase1a": config.phase1aSteps,
        "phase1b": config.phase1bSteps,
        "phase2": config.phase2Steps,
        "phase3": config.phase3Steps,
    }
    if phase not in steps_by_phase:
        raise ValueError(f"Unknown phase: {phase}")
    total_steps = steps_by_phase[phase]

    run_name = f"{config.environmentName}_{config.runName}"
    checkpoint_to_load = checkpointPath(config)
    # Metrics and plots are Phase-scoped because every Phase has a different
    # metrics schema; writing them to a single ``<run_name>.csv`` mixes headers
    # and corrupts the file.  Checkpoints keep the legacy ``<run_name>_<suffix>``
    # naming (e.g. ``..._320k`` and ``..._320k_final``).
    metrics_filename = os.path.join(config.folderNames.metricsFolder, f"{run_name}_{phase}")
    plot_filename = os.path.join(config.folderNames.plotsFolder, f"{run_name}_{phase}")
    checkpoint_filename_base = os.path.join(config.folderNames.checkpointsFolder, run_name)
    video_filename_base = os.path.join(config.folderNames.videosFolder, run_name)
    ensureParentFolders(metrics_filename, plot_filename, checkpoint_filename_base, video_filename_base)

    if mode == "offline":
        observation_shape, action_size = ReplayBuffer.peekOffline(config.offlineDataPath)
        action_low = list(getattr(config, "actionLow", [-1.0] * action_size))
        action_high = list(getattr(config, "actionHigh", [1.0] * action_size))
        env = None
    else:
        env = build_pixels_environment(config.environmentName)
        observation_shape, action_size, action_low, action_high = get_env_properties(env)

    evaluate = evaluation_enabled(config, mode)
    env_evaluation = None
    if mode == "online":
        env_evaluation = build_pixels_environment(config.environmentName, render_mode="rgb_array")

    print(f"envProperties: obs {observation_shape}, action size {action_size}, low {action_low}, high {action_high}")

    dreamer = DreamerV4(observation_shape, action_size, action_low, action_high, device, config.dreamer)
    # Report the executed precision/dtype contract for both online and offline.
    dreamer.logPrecisionContract()
    if config.resume:
        if not getattr(config, "checkpointToLoad", ""):
            raise ValueError(
                "resume=True requires a checkpoint to load; set 'checkpointToLoad' "
                "in the config or pass --checkpoint-to-load <suffix>."
            )
        dreamer.loadCheckpoint(checkpoint_to_load)

    if mode == "offline":
        load_offline_dataset(
            dreamer,
            config.offlineDataPath,
            config.dreamer.batchSize,
            config.dreamer.batchLengthLong,
            missing_relevant_policy=getattr(config, "missingRelevantPolicy", "uniform"),
        )
    else:
        dreamer.environmentInteraction(
            env,
            config.episodesBeforeStart,
            seed=config.seed,
            relevant_episode_quantile=getattr(config, "onlineRelevantEpisodeQuantile", 0.75),
        )

    # Phase 1b and Phase 2 require both the uniform and relevant groups to be
    # present in every batch; balanced sampling guarantees that without
    # touching the loss masks (Phase 1a/3 do not use batch relevance).
    balance_relevance = phase in ("phase1b", "phase2")

    for step in range(total_steps):
        sequence_size = (
            config.dreamer.batchLengthLong
            if (step + 1) % config.dreamer.longBatchEvery == 0
            else config.dreamer.batchLengthShort
        )
        batch = dreamer.buffer.sample(
            config.dreamer.batchSize, sequence_size, balance_relevance=balance_relevance
        )

        if phase == "phase1a":
            metrics = dreamer.trainPhase1a(batch)
        elif phase == "phase1b":
            metrics = dreamer.trainPhase1b(batch)
        elif phase == "phase2":
            metrics = dreamer.trainPhase2(batch)
        else:
            metrics = dreamer.trainPhase3(batch)

        dreamer.total_gradient_steps += 1

        if dreamer.total_gradient_steps % config.checkpointInterval == 0 and config.saveCheckpoints:
            suffix = checkpointStepSuffix(dreamer.total_gradient_steps)
            dreamer.saveCheckpoint(f"{checkpoint_filename_base}_{suffix}")
            if evaluate:
                # Evaluation environment is created lazily only when needed.
                if env_evaluation is None:
                    env_evaluation = build_pixels_environment(config.environmentName, render_mode="rgb_array")
                evaluation_score = dreamer.environmentInteraction(
                    env_evaluation,
                    config.numEvaluationEpisodes,
                    seed=config.seed,
                    evaluation=True,
                    save_video=True,
                    filename=f"{video_filename_base}_{suffix}",
                )
                print(f"Saved Checkpoint and Video at {suffix:>6} gradient steps. Evaluation score: {evaluation_score:>8.2f}")
            else:
                print(f"Saved Checkpoint at {suffix:>6} gradient steps.")

        if config.saveMetrics and step % max(1, total_steps // 20) == 0:
            metrics_base = {"envSteps": dreamer.total_env_steps, "gradientSteps": dreamer.total_gradient_steps}
            saveLossesToCSV(metrics_filename, metrics_base | metrics)

    # Critical Phase 2 finalisation and the final checkpoint must happen before
    # the non-critical plot: a plotting failure must never block persistence of
    # the final state.  The plot is left unguarded so its failure still
    # propagates instead of being swallowed.
    if phase == "phase2":
        dreamer.finalizePhase2()
        if dreamer.frozenPrior is None:
            raise RuntimeError(
                "Phase 2 finalize did not create a frozen policy prior; refusing to "
                "produce a checkpoint that Phase 3 cannot load."
            )
        if config.saveCheckpoints:
            final_suffix = checkpointStepSuffix(dreamer.total_gradient_steps)
            final_checkpoint_path = f"{checkpoint_filename_base}_{final_suffix}_final"
            # Final publish is no-overwrite: an existing final checkpoint is
            # never replaced (see DreamerV4.saveFinalCheckpoint).
            dreamer.saveFinalCheckpoint(final_checkpoint_path)
            print(
                f"Saved final Phase 2 checkpoint at {final_checkpoint_path}.pth "
                "(frozenPrior is not None)."
            )

    if config.saveMetrics:
        plotMetrics(f"{metrics_filename}", savePath=f"{plot_filename}", title=f"{config.environmentName}_{phase}")


def main(config_file, phase_override=None, mode_override=None, resume_override=None, checkpoint_override=None):
    """Load config, apply CLI overrides (CLI wins over YAML), then train.

    ``checkpoint_override`` is the checkpoint suffix (not a full path); it is
    resolved with the same ``<checkpointsFolder>/<environmentName>_<runName>_<suffix>``
    rule as the YAML ``checkpointToLoad`` field.  Providing a checkpoint without
    an explicit resume flag enables resume.  ``resume_override`` (True/False)
    always takes precedence over the YAML ``resume`` value.
    """
    config = loadConfig(config_file)
    if phase_override is not None:
        config.phase = phase_override
    if checkpoint_override is not None:
        config.checkpointToLoad = checkpoint_override
        if resume_override is None:
            resume_override = True
    if resume_override is not None:
        config.resume = resume_override
    seedEverything(config.seed)
    mode = resolve_mode(config, mode_override)
    run_training(config, mode)


def buildArgParser():
    parser = argparse.ArgumentParser(description="NaturalDreamerV4 training entry point")
    parser.add_argument("--config", type=str, default="car-racing-v4.yml")
    parser.add_argument("--phase", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None, choices=VALID_MODES)
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=None,
        help="Force resume from a checkpoint (overrides YAML 'resume').",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=None,
        help="Force-disable resume (overrides YAML 'resume').",
    )
    parser.add_argument(
        "--checkpoint-to-load",
        dest="checkpoint_to_load",
        type=str,
        default=None,
        help=(
            "Checkpoint suffix to load, resolved as "
            "<checkpointsFolder>/<environmentName>_<runName>_<suffix>.pth "
            "(overrides YAML 'checkpointToLoad'; implies --resume unless "
            "--no-resume is given)."
        ),
    )
    return parser


if __name__ == "__main__":
    args = buildArgParser().parse_args()
    main(args.config, args.phase, args.mode, args.resume, args.checkpoint_to_load)
