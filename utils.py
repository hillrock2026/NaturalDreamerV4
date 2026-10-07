import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import yaml
import os
import tempfile
import attridict
import csv
import pandas as pd
import plotly.graph_objects as pgo


def seedEverything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def findFile(filename):
    currentDir = os.getcwd()
    for root, dirs, files in os.walk(currentDir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"File '{filename}' not found in subdirectories of {currentDir}")


def loadConfig(config_path):
    if not config_path.endswith(".yml"):
        config_path += ".yml"
    config_path = findFile(config_path)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return attridict(config)


def getEnvProperties(env):
    import gymnasium as gym
    observationShape = env.observation_space.shape
    if isinstance(env.action_space, gym.spaces.Discrete):
        discreteActionBool = True
        actionSize = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        discreteActionBool = False
        actionSize = env.action_space.shape[0]
    else:
        raise Exception
    return observationShape, discreteActionBool, actionSize


def saveLossesToCSV(filename, metrics):
    """Append one metrics row to a Phase-scoped CSV file.

    The header is written when the file is created and is the schema for every
    later row of that file.  Each Phase has a different metrics schema, so
    callers must pass a per-Phase ``filename``.  If the on-disk header does not
    match the incoming ``metrics`` keys the write is refused (fail fast)
    instead of silently appending values under the wrong column names.
    """
    fileAlreadyExists = os.path.isfile(filename + ".csv")
    incomingHeader = list(metrics.keys())
    if fileAlreadyExists:
        with open(filename + ".csv", mode='r', newline='') as file:
            reader = csv.reader(file)
            try:
                existingHeader = next(reader)
            except StopIteration:
                existingHeader = []
        if existingHeader != incomingHeader:
            raise ValueError(
                f"Refusing to append metrics to '{filename}.csv': existing header "
                f"{existingHeader} does not match incoming keys {incomingHeader}. "
                "Each Phase must keep its own file and schema; do not mix Phases."
            )
    with open(filename + ".csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        if not fileAlreadyExists:
            writer.writerow(incomingHeader)
        writer.writerow(metrics.values())


def plotMetrics(filename, title="", savePath="metricsPlot", window=10):
    if not filename.endswith(".csv"):
        filename += ".csv"
    
    data = pd.read_csv(filename)
    fig = pgo.Figure()

    colors = [
        "gold", "gray", "beige", "blueviolet", "cadetblue",
        "chartreuse", "coral", "cornflowerblue", "crimson", "darkorange",
        "deeppink", "dodgerblue", "forestgreen", "aquamarine", "lightseagreen",
        "lightskyblue", "mediumorchid", "mediumspringgreen", "orangered", "violet"]
    num_colors = len(colors)

    for idx, column in enumerate(data.columns):
        if column in ["envSteps", "gradientSteps"]:
            continue
        
        fig.add_trace(pgo.Scatter(
            x=data["gradientSteps"], y=data[column], mode='lines',
            name=f"{column} (original)",
            line=dict(color='gray', width=1, dash='dot'),
            opacity=0.5, visible='legendonly'))
        
        smoothed_data = data[column].rolling(window=window, min_periods=1).mean()
        fig.add_trace(pgo.Scatter(
            x=data["gradientSteps"], y=smoothed_data, mode='lines',
            name=f"{column} (smoothed)",
            line=dict(color=colors[idx % num_colors], width=2)))
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=30),
            yanchor='top'
        ),
        xaxis=dict(
            title="Gradient Steps",
            showgrid=True,
            zeroline=False,
            position=0
        ),
        yaxis_title="Value",
        template="plotly_dark",
        height=1080,
        width=1920,
        margin=dict(t=60, l=40, r=40, b=40),
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="White",
            borderwidth=2,
            font=dict(size=12)
        )
    )

    if not savePath.endswith(".html"):
        savePath += ".html"
    fig.write_html(savePath)


def sequentialModel1D(inputSize, hiddenSizes, outputSize, activationFunction="Tanh", finishWithActivation=False):
    activationFunction = getattr(nn, activationFunction)()
    layers = []
    currentInputSize = inputSize

    for hiddenSize in hiddenSizes:
        layers.append(nn.Linear(currentInputSize, hiddenSize))
        layers.append(activationFunction)
        currentInputSize = hiddenSize
    
    layers.append(nn.Linear(currentInputSize, outputSize))
    if finishWithActivation:
        layers.append(activationFunction)

    return nn.Sequential(*layers)


def computeLambdaValues(rewards, values, continues, lambda_=0.95):
    returns = torch.zeros_like(rewards)
    bootstrap = values[:, -1]
    for i in reversed(range(rewards.shape[-1])):
        returns[:, i] = rewards[:, i] + continues[:, i] * ((1 - lambda_) * values[:, i] + lambda_ * bootstrap)
        bootstrap = returns[:, i]
    return returns


def ensureParentFolders(*paths):
    for path in paths:
        parentFolder = os.path.dirname(path)
        if parentFolder and not os.path.exists(parentFolder):
            os.makedirs(parentFolder, exist_ok=True)


def fsyncDirectory(directory):
    """Flush a directory's entries to disk so a rename is durable.

    Directory fsync is a no-op on platforms that do not support it (Windows).
    On supported platforms a failure propagates to the caller rather than being
    swallowed.
    """
    if os.name == "nt":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _unlinkQuietly(path):
    """Best-effort removal of a temporary file during error cleanup.

    Only ``OSError`` is caught, and the error is never re-raised: cleanup must
    not mask the original failure that triggered it.
    """
    try:
        os.unlink(path)
    except OSError:
        pass


def atomicWriteFile(path, writeCallback, overwrite=True):
    """Atomically create ``path`` using ``writeCallback(fileHandle)``.

    The callback serializes into a uniquely named temporary file created in the
    *same directory* as ``path`` (same filesystem, so publishing is atomic).
    The temporary file is flushed and fsynced before publishing, and the
    destination directory is fsynced afterwards.

    * ``overwrite=True`` publishes with ``os.replace`` (unconditional replace,
      the historical checkpoint semantics).
    * ``overwrite=False`` publishes with ``os.link`` + unlink, so creating
      ``path`` is an atomic *no-replace* operation: if ``path`` already exists
      (even if it was created concurrently after the early existence check) the
      link fails with ``FileExistsError`` and the existing target is never
      touched.

    On any failure before the publish, the exception propagates, the temporary
    file is removed, and the destination is never created, truncated or
    deleted.  A ``writeCallback`` must write the complete file; it must not
    close the handle (the caller owns it).
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    if not overwrite and os.path.exists(path):
        raise FileExistsError(
            f"Refusing to overwrite existing file '{path}' (overwrite=False)."
        )
    fd, tempPath = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            writeCallback(handle)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(tempPath, path)
        else:
            # Atomic no-replace: ``os.link`` fails with ``FileExistsError`` if
            # ``path`` exists, closing the check-then-replace TOCTOU window.
            os.link(tempPath, path)
            _unlinkQuietly(tempPath)
    except BaseException:
        # Publish never happened (or the publish itself failed): drop the temp
        # file and let the original exception surface untouched.
        _unlinkQuietly(tempPath)
        raise
    # Past this point the target is complete and valid.  The directory fsync is
    # a post-publish durability step; if it fails the error propagates but the
    # already-published file is intentionally left intact (unpublishing it
    # would be destructive and race with readers).
    fsyncDirectory(directory)


def publishFileNoReplace(source, target):
    """Atomically publish ``source`` at ``target`` without overwriting.

    ``os.link`` creates ``target`` atomically and fails with
    ``FileExistsError`` if ``target`` already exists (including the case where
    another process creates it concurrently), so there is no check-then-replace
    window.  On success the source name is unlinked -- ``source`` and ``target``
    share one inode -- and the destination directory is fsynced so the publish
    (and the unlink) are durable.

    ``source`` is treated as a temporary candidate owned by the caller and is
    removed on a refused publish (``FileExistsError``) so a refusal leaves no
    temporary file behind.  On any other failure the source is left in place for
    the caller to inspect/clean, and the existing target is never deleted,
    truncated or overwritten.
    """
    directory = os.path.dirname(os.path.abspath(target)) or "."
    try:
        os.link(source, target)
    except FileExistsError:
        _unlinkQuietly(source)
        raise
    _unlinkQuietly(source)
    fsyncDirectory(directory)
