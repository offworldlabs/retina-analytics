"""Vulture dead-code whitelist.

Read by tools/check-dead-code.sh. Every name here is one vulture reports as
dead but which must not be deleted — or which nobody has decided about yet.
The distinction matters, so the two live in separate sections.

Add to CONTRACTS only when the name is genuinely referenced by something
vulture cannot see: a framework calling in, a wire format, a config key. Real
dead code should be deleted, not whitelisted.

The UNREVIEWED section is a backlog, not an exemption. Each entry is code that
appears genuinely unreachable and needs a decision — delete it, or wire up
whatever was left unfinished. The gate is green with these listed so that it
starts catching NEW dead code immediately; working through them is separate.
"""
# ruff: noqa: B018, F821
# B018 — bare-name expressions are how vulture whitelists work.
# F821 — these names are defined in other modules; only vulture reads this file.

_ = type("_", (), {})()

# ── Contracts: referenced by something vulture cannot see ─────────────────────
# Consumed by Tower-Finder's backend across the submodule boundary (verified
# against its origin/main). Vulture scans this library alone, so cross-repo
# callers are invisible and these read as dead.

# Status counter read via getattr(assoc, "track_pairs_gated", 0) in
# backend/routes/analytics.py (and backend/scripts/association_bench.py).
_.track_pairs_gated
# Called from backend/core/state.py.
_.coverage_limit_for
# Called from backend/services/tasks/analytics_refresh.py.
_.coverage_digest
# Called from backend/services/frame_processor.py.
_.record_node_tracks
# Status counter read via getattr(_a, "anchored_inputs_emitted", 0) in
# backend/routes/analytics.py, and directly in backend/routes/test.py.
_.anchored_inputs_emitted
# Called from backend/services/tasks/solver.py and
# backend/services/tasks/analytics_refresh.py.
_.learned_fov_for
# Called from backend/services/tasks/analytics_refresh.py.
_.record_negative_event
# Imported and re-exported by backend/services/geo.py.
M_PER_DEG_LAT

# ── UNREVIEWED: appears dead, needs a decision (delete, or finish wiring) ──────
# Per-class test-reset hooks added in 7f82355; no caller remains in src or
# tests, so they are orphaned rather than dynamically referenced.
#   src/retina_analytics/association.py:819, src/retina_analytics/manager.py:50  (unused method)
_._reset_for_tests
# Fields on the AssociationCandidate of the superseded detection-level path
# (detection_association.py, preserved as a bench module): set at construction,
# never read anywhere in the estate.
#   src/retina_analytics/detection_association.py:45,46  (unused variable)
det_a_idx
det_b_idx
#   src/retina_analytics/detection_association.py:54,55  (unused variable)
grid_delay_a
grid_delay_b
# Write-only instrumentation counters on the same bench module: accumulated on
# DetectionAssociator but never read or emitted.
#   src/retina_analytics/detection_association.py:312/371, 313/372, 315/377  (unused attribute)
_.frame_skew_ms_total
_.frame_skew_samples
_.frame_sync_rejects
# Appears unreachable; carried over from main's backlog at its lineage location.
#   src/retina_analytics/manager.py:483  (unused method)
_.maybe_auto_save
