"""The four re-ranking objectives (Task 3.1, Track E): the ``gain(i | L)`` of ``greedy.py``.

Each objective reuses Track B's metric code, so the re-ranker optimises exactly the number that is
reported (definitions in ``project/metrics/README.md``). L is the list built so far, i a candidate.

MMR (diversity, W2S2 Diversity slides 10-11) -> ILD@K up
    gain = mean genre Jaccard distance between i and the items of L (0 for the first slot).
    Adding i adds sum_{j in L} d(i, j) to the list's total pairwise distance; dividing by |L| keeps
    the gain in [0, 1] as the list grows. Classic MMR uses the minimum distance instead; the mean is
    i's contribution to ILD, the metric we report.

Calibrated (calibration, Steck 2018; W3S1 Calibration slides 12-18) -> MC_KL@K down
    gain = -KL(p(g|u) || q~(g | L + i)),  q~ = (1 - alpha) q + alpha p,  alpha = 0.01
    p = genre mix of the user's training interactions, q = genre mix of the list (uniform weights).
    Maximising (1 - lam) rel(i) + lam gain(i | L) at every step is the slide's greedy solution of
    argmax (1 - lam) sum rel - lam KL.

PopCalibrated (user-side fairness, Abdollahpouri et al. UMAP 2021; W3S2 Fairness slides 39-40) -> UPD@K down
    gain = -JSD(P(H_u), P(L + i)), P = share of head / mid / tail / unseen items.
    Every user gets the popularity mix of their own profile, so niche users are not served the same
    blockbusters as mainstream users.

xQuAD (item-side fairness, Abdollahpouri et al. FLAIRS 2019; W3S2 Fairness slides 41-43, 48) -> TailShare@K up
    gain = sum_c P(c|u) [i in c] (1 - share of L in c),  c in {short head, long tail}
    Long tail = tail + unseen items (the definition of TailShare), short head = head + mid.
    "Smooth" xQuAD: a category's bonus shrinks as it fills the list, instead of vanishing after one
    item. Choice: P(c|u) = 1/2 by default, so every list makes room for long-tail items (exposure
    for the items, the item-side goal); ``personalised=True`` uses the user's own long-tail share,
    as in the paper, which makes it close to PopCalibrated.
"""
from __future__ import annotations

from typing import Callable, Dict, Type

import numpy as np

from project.metrics import EvaluationContext, calibration, diversity, lookups, popularity


class Objective:
    """A beyond-accuracy goal: catalogue-level arrays (``prepare``) and one state per user."""

    name: str = ""
    target_metric: str = ""  # metric key in project.metrics.METRIC_INFO that the objective improves

    def prepare(self, ctx: EvaluationContext) -> None:
        """Pre-compute what every user needs; a no-op if ``ctx`` is unchanged."""
        if getattr(self, "_ctx", None) is not ctx:
            self._ctx = ctx
            self._prepare(ctx)

    def _prepare(self, ctx: EvaluationContext) -> None:
        pass

    def user_state(self, user_row: int, cand: np.ndarray):
        """State of user ``ctx.users[user_row]`` for candidates ``cand`` (catalogue indices, original order)."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ----------------------------------------------------------------------------- diversity
class _MeanDistance:
    def __init__(self, dist: np.ndarray):
        self.dist = dist  # (n, n) distances between the candidates
        self.total = np.zeros(len(dist))
        self.size = 0

    def gains(self) -> np.ndarray:
        return self.total / self.size if self.size else np.zeros(len(self.total))

    def add(self, j: int) -> None:
        self.total += self.dist[:, j]
        self.size += 1


class MMR(Objective):
    name, target_metric = "MMR", "ild"

    def user_state(self, user_row, cand):
        return _MeanDistance(diversity.jaccard_distance_matrix(self._ctx.genres[cand]).astype(float))


# ------------------------------------------------------- calibration (genres, popularity)
class _DistributionMatch:
    """gain = -divergence(target, q(L + i)), q = mean item distribution of the list."""

    def __init__(self, target: np.ndarray, item_dist: np.ndarray, divergence: Callable):
        self.target, self.item_dist, self.divergence = target, item_dist, divergence
        self.mass = np.zeros(item_dist.shape[1])
        self.size = 0

    def gains(self) -> np.ndarray:
        if np.isnan(self.target).any():  # no profile: nothing to match, keep the model's order
            return np.zeros(len(self.item_dist))
        q = (self.mass + self.item_dist) / (self.size + 1)
        return -self.divergence(np.broadcast_to(self.target, q.shape), q)

    def add(self, j: int) -> None:
        self.mass += self.item_dist[j]
        self.size += 1


class Calibrated(Objective):
    name, target_metric = "Calibrated", "miscalibration"

    def __init__(self, alpha: float = calibration.ALPHA):
        self.alpha = alpha

    def _prepare(self, ctx):
        self.profiles = calibration.profile(ctx.train, ctx.p_gi)  # (U, G) p(g|u) of the training interactions

    def _kl(self, p, q):
        return calibration.kl_miscalibration(p, q, self.alpha)

    def user_state(self, user_row, cand):
        return _DistributionMatch(self.profiles[user_row], self._ctx.p_gi[cand], self._kl)


class PopCalibrated(Objective):
    name, target_metric = "PopCalibrated", "upd"

    def _prepare(self, ctx):
        n_groups = len(lookups.POPULARITY_GROUPS)
        self.one_hot = np.eye(n_groups)[ctx.item_groups]
        self.profiles = popularity.group_distribution_of_histories(ctx.train, ctx.item_groups, n_groups)

    def user_state(self, user_row, cand):
        return _DistributionMatch(self.profiles[user_row], self.one_hot[cand], popularity.jensen_shannon)


# --------------------------------------------------------------------- item-side fairness
class _CategoryCoverage:
    """Smooth xQuAD with two categories: long tail and short head."""

    def __init__(self, in_tail: np.ndarray, tail_weight: float):
        self.in_tail, self.tail_weight = in_tail, tail_weight
        self.size = self.tail = 0

    def gains(self) -> np.ndarray:
        tail_share = self.tail / self.size if self.size else 0.0
        head_share = 1.0 - tail_share if self.size else 0.0
        return np.where(self.in_tail, self.tail_weight * (1 - tail_share), (1 - self.tail_weight) * (1 - head_share))

    def add(self, j: int) -> None:
        self.size += 1
        self.tail += int(self.in_tail[j])


class XQuAD(Objective):
    name, target_metric = "xQuAD", "tailshare"

    def __init__(self, personalised: bool = False):
        self.personalised = personalised

    def _prepare(self, ctx):
        self.long_tail = np.isin(ctx.item_groups, [lookups.TAIL, lookups.UNSEEN])
        if self.personalised:
            n = ctx.train.sum(axis=1)
            self.tail_weight = np.divide((ctx.train & self.long_tail).sum(axis=1), n,
                                         out=np.full(len(n), 0.5), where=n > 0)
        else:
            self.tail_weight = np.full(len(ctx.users), 0.5)

    def user_state(self, user_row, cand):
        return _CategoryCoverage(self.long_tail[cand], self.tail_weight[user_row])

    def __repr__(self) -> str:
        return f"XQuAD(personalised={self.personalised})"


OBJECTIVES: Dict[str, Type[Objective]] = {o.name: o for o in (MMR, Calibrated, PopCalibrated, XQuAD)}


def make_objective(name: str) -> Objective:
    if name not in OBJECTIVES:
        raise ValueError(f"unknown objective {name!r}; choose from {list(OBJECTIVES)}")
    return OBJECTIVES[name]()
