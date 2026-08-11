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
# (none)

# ── UNREVIEWED: appears dead, needs a decision (delete, or finish wiring) ──────
# TODO: no reference found anywhere in the estate
#   src/retina_analytics/association.py:33  (unused variable)
C_KM_S
# TODO: no reference found anywhere in the estate
#   src/retina_analytics/association.py:135  (unused variable)
det_a_idx
# TODO: no reference found anywhere in the estate
#   src/retina_analytics/association.py:136  (unused variable)
det_b_idx
# TODO: no reference found anywhere in the estate
#   src/retina_analytics/association.py:97  (unused variable)
fc_hz
# TODO: no reference found anywhere in the estate
#   src/retina_analytics/manager.py:199  (unused method)
_.get_all_summaries
# TODO: no reference found anywhere in the estate
#   src/retina_analytics/manager.py:311  (unused method)
_.maybe_auto_save
