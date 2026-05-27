"""Failure replay utilities for medium-level dual-energy training.

The replay buffer stores small collated mini-batches, not raw dataset indices.  This
keeps it independent of a particular Dataset implementation and works with the
variable-obstacle padding used by ``NavMaximinLocalPatches``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional
import random

import torch


def _batch_size(batch: Mapping[str, Any]) -> int:
    if "goal_map" in batch and torch.is_tensor(batch["goal_map"]):
        return int(batch["goal_map"].shape[0])
    if "meta" in batch and isinstance(batch["meta"], Mapping):
        for v in batch["meta"].values():
            if torch.is_tensor(v) and v.ndim >= 1:
                return int(v.shape[0])
    raise ValueError("could not infer batch size")


def _clone_value(v: Any, idx: Optional[torch.Tensor], B: int) -> Any:
    if torch.is_tensor(v):
        vv = v.detach().cpu()
        if idx is not None and vv.ndim >= 1 and int(vv.shape[0]) == B:
            vv = vv.index_select(0, idx.cpu())
        return vv.clone()
    if isinstance(v, Mapping):
        return {k: _clone_value(x, idx, B) for k, x in v.items()}
    if isinstance(v, list):
        return list(v)
    return v


def clone_batch_to_cpu(batch: Mapping[str, Any], idx: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """Clone a collated batch, optionally selecting sample indices on dim 0."""
    B = _batch_size(batch)
    return {k: _clone_value(v, idx, B) for k, v in batch.items()}


@dataclass
class ReplayEntry:
    batch: Dict[str, Any]
    score_mean: float
    tag_counts: Dict[str, int]


class FailureReplayBuffer:
    """Capacity-limited replay buffer for failure-heavy local snapshots."""

    def __init__(self, capacity: int = 256, seed: int = 0):
        self.capacity = int(max(0, capacity))
        self.rng = random.Random(int(seed))
        self.entries: List[ReplayEntry] = []
        self.seen = 0

    def __len__(self) -> int:
        return len(self.entries)

    def add_from_scores(
        self,
        batch: Mapping[str, Any],
        scores: torch.Tensor,
        tags: Mapping[str, torch.Tensor],
        min_score: float = 0.05,
        topk: int = 4,
    ) -> int:
        """Store the top failure samples from a normal training mini-batch.

        Parameters
        ----------
        batch:
            Original collated CPU batch from the DataLoader.
        scores:
            Per-sample nonnegative failure scores; larger means more useful replay.
        tags:
            Per-sample boolean/float tag tensors for logging.
        min_score:
            Minimum score for a sample to be retained.
        topk:
            Maximum retained samples from this mini-batch.
        """
        if self.capacity <= 0:
            return 0
        s = scores.detach().float().cpu().flatten()
        if s.numel() == 0:
            return 0
        k = min(int(max(1, topk)), int(s.numel()))
        vals, idx = torch.topk(s, k=k)
        keep = vals >= float(min_score)
        if not bool(keep.any()):
            return 0
        idx = idx[keep]
        vals = vals[keep]
        tag_counts: Dict[str, int] = {}
        for name, mask in tags.items():
            m = mask.detach().cpu().flatten()
            if m.numel() == s.numel():
                tag_counts[name] = int((m.index_select(0, idx) > 0).sum().item())
        entry = ReplayEntry(
            batch=clone_batch_to_cpu(batch, idx),
            score_mean=float(vals.mean().item()),
            tag_counts=tag_counts,
        )
        self._append(entry)
        return int(idx.numel())

    def _append(self, entry: ReplayEntry) -> None:
        self.seen += 1
        if len(self.entries) < self.capacity:
            self.entries.append(entry)
            return
        # Reservoir-style replacement biased toward recent hard failures.
        j = self.rng.randrange(self.seen)
        if j < self.capacity:
            self.entries[j] = entry

    def sample(self) -> Optional[Dict[str, Any]]:
        if not self.entries:
            return None
        # Harder entries are sampled more often without making old failures impossible.
        weights = [max(1e-4, e.score_mean) for e in self.entries]
        entry = self.rng.choices(self.entries, weights=weights, k=1)[0]
        return clone_batch_to_cpu(entry.batch)

    def tag_summary(self) -> Dict[str, float]:
        if not self.entries:
            return {}
        total: Dict[str, int] = {}
        for e in self.entries:
            for k, v in e.tag_counts.items():
                total[k] = total.get(k, 0) + int(v)
        denom = max(1, len(self.entries))
        return {f"replay_tag_{k}": float(v) / denom for k, v in total.items()}
