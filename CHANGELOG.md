# Changelog

## 0.1.0 - 2026-04-01

### Added
- Package scaffold as repository-root modules (`cli`, `config`, `data`, `inference`, `model`, `training`, `utils`, `validation`).
- Stage scheduler, loss computation module, checkpoint manager, metrics accumulator, trainer engine, and inference runner.
- Data subsystem ground up build: mask handling, tokenization, dataset base classes, collator, LR generation, and deterministic split pipeline.
- End-to-end CLI workflows for train/test/preprocess/split.
- Runtime and peak-memory telemetry in train/test/preprocess/split summaries.
- Parity validation module and `kst-parity` CLI command for baseline-vs-candidate acceptance gating.
- Extensive unit/integration/smoke tests, including CLI smoke coverage.

### Changed
- Positional encoding and tensor utilities now use input/runtime device instead of hardcoded CUDA allocations.
- Metric key naming is collision-safe across split/stage/family/metric dimensions.
