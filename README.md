# retina-analytics

Node analytics for the RETINA passive radar network: trust, reputation,
coverage and cross-node analysis. It is a library with no service of its own;
retina-server uses it as a git submodule under `libs/retina-analytics`.

## Modules

All modules are in `src/retina_analytics/`.

| Module | Purpose |
|---|---|
| `manager.py` | `NodeAnalyticsManager`, the central aggregator for all connected nodes |
| `association.py` | Inter-node association gate: decides which observations from different nodes describe the same target, by testing bistatic delay and Doppler against pre-computed per-pair gates |
| `detection_association.py` | The earlier detection-level association, kept as a baseline for benchmarks |
| `empirical_coverage.py` | Detection-area polygons built from known-position calibration points (ADS-B or multi-node solutions) |
| `coverage.py` | Historical ADS-B-validated coverage map accumulation, with persistent storage |
| `detection_area.py` | Detection-area characterisation from observed delay and Doppler bounds |
| `cross_node.py` | Cross-node comparison and coverage suggestions |
| `trust.py` | ADS-B correlation trust scoring per node |
| `reputation.py` | Bad-actor detection and reputation tracking per node |
| `metrics.py` | Per-node availability, SNR and track-quality metrics |
| `availability.py` | Which minutes of the trailing week a node was available |
| `constants.py` | Shared constants and helpers |

## Development

Requires Python 3.11 or later.

```bash
pip install -e '.[dev]'
pytest
```

CI (`.github/workflows/ci.yml`) also runs `pre-commit run --all-files`, which
covers ruff and the dead-code check.

## License

Copyright (C) 2026 Offworld Labs

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero
General Public License for more details.

The full licence text is in [LICENSE](LICENSE).
