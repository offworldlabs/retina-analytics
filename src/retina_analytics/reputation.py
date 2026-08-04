"""Bad actor detection and reputation tracking per node."""

import time
from dataclasses import dataclass, field


@dataclass
class NodeReputation:
    """Tracks reputation and handles bad actor detection/blocking for a node."""
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
        escalation to a block is the intended defence
        (test_bad_actor_gets_blocked pins it).
        """
        was_active = self._condition_active.get(key, False)
        self._condition_active[key] = active
        if active and not was_active:
            self.apply_penalty(amount, reason)

    def apply_penalty(self, amount: float, reason: str):
        self.reputation = max(0.0, self.reputation - amount)
        self.penalties.append({
            "time": time.time(),
            "amount": amount,
            "reason": reason,
            "reputation_after": self.reputation,
        })
        if len(self.penalties) > self.max_penalties:
            self.penalties = self.penalties[-self.max_penalties:]
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
                "heartbeat_stale", gap > self.min_heartbeat_interval_s,
                0.1, f"Heartbeat stale: {gap:.0f}s",
            )

    def evaluate_detection_rate(self, avg_det_per_frame: float):
        # Per-pass on purpose: sustained flooding should escalate (see
        # evaluate_trust).
        if avg_det_per_frame > self.max_detections_per_frame:
            self.apply_penalty(0.05, f"High detection rate: {avg_det_per_frame:.1f}/frame")

    def evaluate_neighbour_consistency(self, overlap_ratio: float,
                                       neighbour_trust: float,
                                       neighbour_id: str = ""):
        # Keyed per neighbour: the caller loops over every pair each cycle,
        # so an un-keyed condition would both mask distinct neighbours and
        # re-penalise the same disagreement every pass.
        self._penalize_condition(
            f"neighbour_inconsistent:{neighbour_id}",
            neighbour_trust > 0.7 and overlap_ratio < 0.05,
            0.08,
            f"Inconsistent with trusted neighbour {neighbour_id or '?'} "
            f"(overlap={overlap_ratio:.2f})",
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
        }
