# Contributing

## Setup for development

```bash
git clone https://github.com/Aditya-Khankar/cognidrift
cd cognidrift
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/
pytest tests/ -v          # verbose
pytest tests/ --cov       # with coverage
```

All tests must pass before submitting a pull request.

## Project structure

```
cognidrift/
├── cognidrift/
│   ├── recorder/       # Trace ingestion and format adapters
│   ├── detectors/      # Behavioral bias detectors (core logic)
│   ├── metrics/        # Efficiency, exploration, recovery metrics
│   ├── report/         # Report builder and visualization
│   └── utils/          # Validation, path safety, format detection
├── runs/
│   └── sample_runs/    # Pre-generated trace files for testing
└── tests/              # One test file per module
```

## Adding a new detector

1. Create `cognidrift/detectors/your_detector.py`
2. Subclass `BaseDetector` from `cognidrift/detectors/base.py`
3. Implement `detect(self, trace: AgentTrace) -> DetectorResult`
4. Add to `DETECTORS` list in `cognidrift/report/builder.py`
5. Add tests in `tests/test_your_detector.py`
6. Update the detectors table in `README.md`

## Adding a new format adapter

1. Create your adapter class in `cognidrift/recorder/format_adapters.py`
2. Register it in the `get_adapter()` function
3. Add format fingerprint to `cognidrift/utils/format_detect.py`
4. Add tests in `tests/test_adapters.py`
5. Add to supported formats list in `README.md`

## Trace format

All adapters must output an `AgentTrace` with `AgentEvent` objects.
See `cognidrift/recorder/trace_recorder.py` for the canonical schema.

## Code style

```bash
black cognidrift/
ruff check cognidrift/
```

Both must pass with no errors before committing.

## Submitting changes

- **Consistency**: Maintain the `Cognidrift` (prose) vs `cognidrift` (CLI/code) branding.
- **Simplicity**: Prioritize zero-dependency or low-dependency solutions.
- **Parallel Mode Stability**: New detectors must be tested against large trace batches to ensure they are thread/process safe.
- **Threshold Awareness**: New detectors must pull their thresholds from the global config via `builder.py`.
- **Minimalism**: Keep the main branch commit history clean and professional.
