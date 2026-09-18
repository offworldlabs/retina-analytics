"""Bad actor detection and reputation tracking per node."""

import math
import time
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class NodeReputation:
    """Tracks reputation and handles bad actor detection/blocking for a node."""

    # Multiplier applied to every penalty, process-wide — see
    # set_penalty_scale.  A ClassVar, NOT a dataclass field, on purpose: the
    # backend persists reputations with dataclasses.asdict() and restores them
    # with NodeReputation(**saved), so a real field would round-trip through
    # snapshots (freezing an operational stance into saved state) and older
    # snapshots would fail to construct.  Tests may still override it per
    # instance (rep.penalty_scale = 1.0).
    penalty_scale: ClassVar[float] = 1.0

    node_id: str
    reputation: float = 1.0
    blocked: bool = False
    block_reason: str = ""
    trust_warn_threshold: float = 0.3
    trust_block_threshold: float = 0.1
    reputation_block_threshold: float = 0.2
    penalties: list[dict] = field(default_factory=list)
    max_penalties: int = 100
    max_detections_per_frame: float = 50.0
    min_heartbeat_interval_s: float = 300.0
    # Which named conditions are currently active — see _penalize_condition.
    _condition_active: dict = field(default_factory=dict)

    def _penalize_condition(self, key: str, active: bool, amount: float, reason: str):
        """Penalise the *onset* of a condition, not every pass it persists.

        Used for conditions that are absences or static configuration facts
        (stale heartbeat, geometric disagreement with a neighbour), which the
        30 s evaluation cadence re-observes unchanged every cycle.  Per-call
        penalties compounded there: a node that merely went quiet crossed the
        block threshold in ~8 cycles, and unblock() restored it to 0.3 — one
        penalty above re-blocking, with rewards a no-op while blocked.

        Deliberately NOT used for trust and detection-rate penalties: those
        are evidence of bad *data* actively being submitted, and repeated
        escalation to a block is the intended defence whenever downrating is
        switched on at all (test_bad_actor_gets_blocked pins that shape, with
        penalty_scale at its library default of 1.0).  Orthogonal to this:
        set_penalty_scale can turn every downrating path off, escalation
        included — see apply_penalty.
        """
        was_active = self._condition_active.get(key, False)
        self._condition_active[key] = active
        if active and not was_active:
            self.apply_penalty(amount, reason)

    def apply_penalty(self, amount: float, reason: str):
        """Lower the reputation by ``amount`` scaled by penalty_scale.

        Every downrating source in the estate funnels through here (trust,
        heartbeat, detection rate, neighbour consistency, and the backend's
        direct ADS-B cross-validation call), so the scale is the single place
        that decides whether reputations can fall at all.
        """
        effective = amount * self.penalty_scale
        if effective <= 0:
            # Scale at zero: the trust/heartbeat/neighbour paths still compute
            # and report their findings, only the downrating stops.  Nothing is
            # recorded either — "no new penalties" is the operator's check that
            # the switch is on, so a log of zero-amount entries would make
            # n_penalties lie about it.
            return
        self.reputation = max(0.0, self.reputation - effective)
        entry = {
            "time": time.time(),
            "amount": effective,
            "reason": reason,
            "reputation_after": self.reputation,
        }
        if effective != amount:
            entry["base_amount"] = amount
        self.penalties.append(entry)
        if len(self.penalties) > self.max_penalties:
            self.penalties = self.penalties[-self.max_penalties :]
        if self.reputation < self.reputation_block_threshold and not self.blocked:
            self.blocked = True
            self.block_reason = f"Reputation {self.reputation:.2f} below threshold"

    def apply_reward(self, amount: float):
        if not self.blocked:
            self.reputation = min(1.0, self.reputation + amount)

    def evaluate_trust(self, trust_score: float):
        # Per-pass on purpose: low trust means bad data is being submitted
        # right now, and escalation to a block is the point.
        if trust_score < self.trust_block_threshold:
            self.apply_penalty(0.15, f"Trust score critically low: {trust_score:.3f}")
        elif trust_score < self.trust_warn_threshold:
            self.apply_penalty(0.05, f"Trust score low: {trust_score:.3f}")
        elif trust_score > 0.7:
            self.apply_reward(0.01)

    def evaluate_heartbeat(self, last_heartbeat: float):
        if last_heartbeat > 0:
            gap = time.time() - last_heartbeat
            self._penalize_condition(
                "heartbeat_stale",
                gap > self.min_heartbeat_interval_s,
                0.1,
                f"Heartbeat stale: {gap:.0f}s",
            )

    def evaluate_detection_rate(self, avg_det_per_frame: float):
        # Per-pass on purpose: sustained flooding should escalate (see
        # evaluate_trust).
        if avg_det_per_frame > self.max_detections_per_frame:
            self.apply_penalty(0.05, f"High detection rate: {avg_det_per_frame:.1f}/frame")

    def evaluate_neighbour_consistency(self, overlap_ratio: float, neighbour_trust: float, neighbour_id: str = ""):
        # Keyed per neighbour: the caller loops over every pair each cycle,
        # so an un-keyed condition would both mask distinct neighbours and
        # re-penalise the same disagreement every pass.
        self._penalize_condition(
            f"neighbour_inconsistent:{neighbour_id}",
            neighbour_trust > 0.7 and overlap_ratio < 0.05,
            0.08,
            f"Inconsistent with trusted neighbour {neighbour_id or '?'} (overlap={overlap_ratio:.2f})",
        )

    def unblock(self):
        self.blocked = False
        self.block_reason = ""
        self.reputation = 0.3

    def summary(self) -> dict:
        return {
            "node_id": self.node_id,
            "reputation": round(self.reputation, 4),
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "n_penalties": len(self.penalties),
            "recent_penalties": self.penalties[-5:] if self.penalties else [],
            # Published so /api/radar/analytics shows whether downrating is on
            # at all: zero here explains a node whose reputation never moves.
            "penalty_scale": self.penalty_scale,
        }


def set_penalty_scale(scale: float) -> None:
    """Set the one multiplier every reputation-lowering path routes through.

    Process-wide (a ClassVar on NodeReputation).  The retina-server backend
    sets it from the environment (REPUTATION_PENALTY_SCALE) at startup and
    currently defaults it to 0: a temporary stance while the trust input is a
    single claim residual per node, after a real node was permanently blocked
    off one out-of-threshold sample.  At 0 no path here can lower a reputation
    or block a node — trust, heartbeat, detection-rate, neighbour-consistency
    and the backend's ADS-B cross-validation penalty all become no-ops, and
    nothing is recorded in `penalties`.  Rewards are untouched, so a node can
    still climb back to 1.0.  The library default stays 1.0.
    """
    try:
        value = float(scale)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"penalty_scale must be a finite float >= 0, got {scale!r}") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"penalty_scale must be a finite float >= 0, got {scale!r}")
    NodeReputation.penalty_scale = value
