import os

import attridict
import numpy as np
import torch


class ReplayBuffer:
    """Ring buffer for offline/online transition data with relevant flags.

    Episode boundaries
    ------------------
    ``done == 1`` marks the last transition of a trajectory segment and is used
    as an episode boundary.  A ``done`` may come from the environment
    (terminated/truncated) or from a deliberately capped data-collection
    segment (for example the 480-step offline fragments).  ``sample`` never
    returns a contiguous sequence that crosses one of these boundaries.
    """

    def __init__(self, observation_shape, action_size, config, device):
        self.config = config
        self.device = device
        self.capacity = int(config.capacity)
        self.observation_shape = tuple(observation_shape)

        self.observations = np.empty((self.capacity, *self.observation_shape), dtype=np.float32)
        self.nextObservations = np.empty((self.capacity, *self.observation_shape), dtype=np.float32)
        self.actions = np.empty((self.capacity, action_size), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.dones = np.empty((self.capacity, 1), dtype=np.float32)
        self.relevant = np.zeros((self.capacity, 1), dtype=np.bool_)

        self.buffer_index = 0
        self.full = False
        self.missing_relevant = False

        # Episode-boundary-aware sampling.  ``_version`` changes whenever the
        # content changes, so cached legal start indices are only recomputed
        # once per content revision (not on every ``sample`` call).
        self._version = 0
        self._valid_start_cache = {}
        self._group_cache = {}

    def __len__(self):
        return self.capacity if self.full else self.buffer_index

    def add(self, observation, action, reward, next_observation, done, relevant=False):
        self.observations[self.buffer_index] = observation
        self.actions[self.buffer_index] = action
        self.rewards[self.buffer_index] = reward
        self.nextObservations[self.buffer_index] = next_observation
        self.dones[self.buffer_index] = float(done)
        self.relevant[self.buffer_index] = bool(relevant)
        self.buffer_index = (self.buffer_index + 1) % self.capacity
        self.full = self.full or self.buffer_index == 0
        self._version += 1

    def _compute_valid_starts(self, sequence_size):
        """Physical start indices whose ``sequence_size`` window never crosses a done.

        A window ``[s, s + sequence_size - 1]`` is legal when none of its
        interior transitions (all but the last) is an episode end.  The last
        transition may be ``done == 1`` because it is the natural segment end.
        """
        n = len(self)
        if sequence_size <= 1:
            count = self.capacity if self.full else n
            return np.arange(count, dtype=np.int64)
        if n < sequence_size:
            return np.empty(0, dtype=np.int64)

        interior = sequence_size - 1
        if self.full:
            completed = self.dones[: self.capacity].reshape(-1) > 0.5
            doubled = np.concatenate([completed, completed])
            cumulative = np.concatenate([[0], np.cumsum(doubled)])
            crossings = cumulative[interior:interior + self.capacity] - cumulative[:self.capacity]
            return np.flatnonzero(crossings == 0).astype(np.int64)

        completed = self.dones[:n].reshape(-1) > 0.5
        cumulative = np.concatenate([[0], np.cumsum(completed)])
        num_starts = n - sequence_size + 1
        crossings = cumulative[interior:interior + num_starts] - cumulative[:num_starts]
        return np.flatnonzero(crossings == 0).astype(np.int64)

    def _valid_starts(self, sequence_size):
        cached = self._valid_start_cache.get(sequence_size)
        if cached is not None and cached[0] == self._version:
            return cached[1]
        starts = self._compute_valid_starts(sequence_size)
        self._valid_start_cache[sequence_size] = (self._version, starts)
        return starts

    def _start_groups(self, sequence_size):
        """Per-start relevance composition of each legal window.

        Returns ``(starts, has_relevant, has_uniform)`` where ``has_relevant[i]``
        is True when the window starting at ``starts[i]`` contains at least one
        relevant transition and ``has_uniform[i]`` when it contains at least one
        uniform transition.  This lets the sampler guarantee that both groups
        are represented in a batch without touching the loss masks.
        """
        cached = self._group_cache.get(sequence_size)
        if cached is not None and cached[0] == self._version:
            return cached[1], cached[2], cached[3]

        starts = self._valid_starts(sequence_size)
        if len(starts) == 0:
            has_relevant = np.empty(0, dtype=bool)
            has_uniform = np.empty(0, dtype=bool)
        else:
            n = len(self)
            flat = self.relevant[: self.capacity if self.full else n].reshape(-1)
            if self.full:
                doubled = np.concatenate([flat, flat])
                cumulative = np.concatenate([[0], np.cumsum(doubled)])
                counts = (
                    cumulative[sequence_size:sequence_size + self.capacity]
                    - cumulative[:self.capacity]
                )
            else:
                cumulative = np.concatenate([[0], np.cumsum(flat)])
                num_windows = n - sequence_size + 1
                counts = (
                    cumulative[sequence_size:sequence_size + num_windows]
                    - cumulative[:num_windows]
                )
            counts = counts[starts]
            has_relevant = counts > 0
            has_uniform = counts < sequence_size

        self._group_cache[sequence_size] = (
            self._version,
            starts,
            has_relevant,
            has_uniform,
        )
        return starts, has_relevant, has_uniform

    def _sample_start_indices(self, batch_size, sequence_size, balance_relevance=False):
        """Return ``batch_size`` legal window start indices.

        With ``balance_relevance=True`` the batch is guaranteed to contain at
        least one relevant-containing and one uniform-containing window whenever
        the buffer has both kinds of window.  Only the *inclusion* is forced;
        the per-transition relevance masks inside the losses are unchanged, and
        when a whole group genuinely does not exist the batch is left as-is so
        the fail-fast ``_masked_mean`` check still fires.
        """
        starts, has_relevant, has_uniform = self._start_groups(sequence_size)
        if len(starts) == 0:
            raise ValueError(
                f"Buffer has {len(self)} transitions but no contiguous window of "
                f"length {sequence_size} stays inside a single episode; every "
                "candidate would cross a done=1 boundary. Reduce sequence_size or "
                "collect longer episodes."
            )

        indices = np.random.randint(0, len(starts), batch_size)
        if balance_relevance:
            if not bool(has_relevant[indices].any()) and bool(has_relevant.any()):
                candidates = np.flatnonzero(has_relevant)
                indices[np.random.randint(0, batch_size)] = candidates[
                    np.random.randint(0, len(candidates))
                ]
            if not bool(has_uniform[indices].any()) and bool(has_uniform.any()):
                candidates = np.flatnonzero(has_uniform)
                indices[np.random.randint(0, batch_size)] = candidates[
                    np.random.randint(0, len(candidates))
                ]
        return starts[indices]

    def sample(self, batch_size, sequence_size, balance_relevance=False):
        n = len(self)
        if sequence_size > self.capacity:
            raise ValueError(
                f"sequence_size={sequence_size} exceeds buffer capacity {self.capacity}."
            )
        if n < sequence_size:
            raise ValueError(
                f"Buffer has {n} transitions but sequence_size={sequence_size} is "
                "required. Not enough data to form even one sequence."
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        sample_starts = self._sample_start_indices(
            batch_size, sequence_size, balance_relevance=balance_relevance
        ).reshape(-1, 1)
        sequence_length = np.arange(sequence_size).reshape(1, -1)
        sample_index = (sample_starts + sequence_length) % self.capacity

        observations = torch.as_tensor(self.observations[sample_index], device=self.device).float()
        next_observations = torch.as_tensor(self.nextObservations[sample_index], device=self.device).float()
        actions = torch.as_tensor(self.actions[sample_index], device=self.device)
        rewards = torch.as_tensor(self.rewards[sample_index], device=self.device)
        dones = torch.as_tensor(self.dones[sample_index], device=self.device)
        relevant = torch.as_tensor(self.relevant[sample_index], device=self.device)

        return attridict({
            "observations": observations,
            "nextObservations": next_observations,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "isRelevant": relevant,
        })

    def validateForSampling(self, batch_size, sequence_size):
        """Raise a clear error if the buffer cannot serve the requested batch."""
        n = len(self)
        if n == 0:
            raise ValueError("Buffer is empty: no offline data has been loaded.")
        if sequence_size > self.capacity:
            raise ValueError(
                f"sequence_size={sequence_size} exceeds buffer capacity {self.capacity}."
            )
        if n < sequence_size:
            raise ValueError(
                f"Insufficient data: buffer has {n} transitions but sequence_size "
                f"={sequence_size} is required."
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if len(self._valid_starts(sequence_size)) == 0:
            raise ValueError(
                f"Insufficient data: buffer has {n} transitions but no contiguous "
                f"window of length {sequence_size} stays inside a single episode. "
                "Every candidate would cross a done=1 boundary."
            )
        return n

    def saveOffline(self, path):
        path = path if path.endswith(".npz") else path + ".npz"
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        n = len(self)
        np.savez_compressed(
            path,
            observations=self.observations[:n],
            nextObservations=self.nextObservations[:n],
            actions=self.actions[:n],
            rewards=self.rewards[:n],
            dones=self.dones[:n],
            relevant=self.relevant[:n],
        )

    @staticmethod
    def peekOffline(path):
        """Read only the shapes needed to construct a model for an offline file."""
        path = path if path.endswith(".npz") else path + ".npz"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Offline dataset not found at: {path}")
        with np.load(path) as data:
            if "observations" not in data or "actions" not in data:
                raise ValueError(
                    f"Offline dataset {path} must contain 'observations' and 'actions' arrays."
                )
            observation_shape = tuple(int(s) for s in data["observations"].shape[1:])
            action_size = int(data["actions"].shape[1])
        return observation_shape, action_size

    def loadOffline(self, path, missing_relevant_policy="uniform"):
        """Load an offline .npz file into the ring buffer.

        ``missing_relevant_policy`` controls what happens when the file has no
        ``relevant`` array:
          * ``"uniform"`` (default): treat the data as ordinary uniform data
            (``relevant=False``), keeping NaturalDreamer data compatible;
          * ``"error"``: raise ``ValueError``.
        """
        if missing_relevant_policy not in ("uniform", "error"):
            raise ValueError(
                f"Unknown missing_relevant_policy '{missing_relevant_policy}'. "
                "Expected 'uniform' or 'error'."
            )
        path = path if path.endswith(".npz") else path + ".npz"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Offline dataset not found at: {path}")
        with np.load(path) as data:
            for key in ("observations", "actions", "rewards"):
                if key not in data:
                    raise ValueError(
                        f"Offline dataset {path} is missing required array '{key}'."
                    )

            n = min(len(data["observations"]), self.capacity)
            if n == 0:
                raise ValueError(f"Offline dataset {path} contains no transitions.")

            self.observations[:n] = data["observations"][:n]
            self.nextObservations[:n] = (
                data["nextObservations"][:n] if "nextObservations" in data else data["observations"][:n]
            )
            self.actions[:n] = data["actions"][:n]
            self.rewards[:n] = data["rewards"][:n]
            self.dones[:n] = data["dones"][:n] if "dones" in data else 0.0

            if "relevant" in data:
                rel = data["relevant"]
                if rel.ndim == 2 and rel.shape[1] == 1:
                    rel = rel.reshape(-1)
                elif rel.ndim != 1:
                    raise ValueError(
                        f"Offline dataset {path} 'relevant' has unsupported shape "
                        f"{rel.shape}; expected (N,) or (N, 1)."
                    )
                if len(rel) < n:
                    raise ValueError(
                        f"Offline dataset {path} 'relevant' has {len(rel)} entries "
                        f"but {n} transitions are required."
                    )
                self.relevant[:n] = rel[:n].astype(bool).reshape(n, 1)
                self.missing_relevant = False
            else:
                self.missing_relevant = True
                if missing_relevant_policy == "error":
                    raise ValueError(
                        f"Offline dataset {path} is missing the 'relevant' field and "
                        "missingRelevantPolicy='error'."
                    )
                self.relevant[:n] = False

        self.buffer_index = n % self.capacity
        self.full = n >= self.capacity
        self._version += 1
        return n
