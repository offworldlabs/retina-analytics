"""Shared test fixtures.

NodeReputation.penalty_scale is a process-wide ClassVar, so a test that moves
it would leak into every test that runs after it (and the leak is silent: the
victims just stop being penalised).  Pin it per test and put it back.
"""

import pytest

from retina_analytics.reputation import NodeReputation, set_penalty_scale


@pytest.fixture(autouse=True)
def penalty_scale_1_0():
    """Run every test with downrating explicitly switched on, then restore.

    1.0 is the library default and the value the penalty/block assertions in
    this suite are written against; the deployment stance (0) is set by the
    backend, not here.  Tests that exercise another scale call
    set_penalty_scale themselves — this fixture undoes it afterwards.
    """
    saved = NodeReputation.penalty_scale
    set_penalty_scale(1.0)
    yield
    NodeReputation.penalty_scale = saved
