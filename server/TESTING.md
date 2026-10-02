# Testing Guide for Shrimply Server

This document describes testing infrastructure, patterns, and conventions for the Shrimply server Python codebase.

## Overview

- **Test Framework**: pytest 8.0+
- **Coverage Tool**: pytest-cov
- **Test Location**: `server/tests/`
- **Source Code**: `server/src/api/`

## Running Tests

### All tests with coverage
```bash
cd server
pytest --cov=api tests/
```

### Specific test file
```bash
pytest tests/test_video_generation.py -v
```

### Specific test
```bash
pytest tests/test_video_generation.py::VideoGenerationChecks::test_request_and_h3_constraints -v
```

### Run only unit tests
```bash
pytest -m unit tests/
```

### Run tests without slow tests
```bash
pytest -m "not slow" tests/
```

### Generate HTML coverage report
```bash
pytest --cov=api --cov-report=html tests/
# Open htmlcov/index.html in browser
```

## Test Markers

Tests can be marked with categories to control execution:

```python
import pytest

@pytest.mark.unit
def test_simple_function():
    pass

@pytest.mark.integration
def test_with_external_service():
    pass

@pytest.mark.slow
def test_model_inference():
    pass
```

## Common Testing Patterns

### 1. Mocking Heavy Dependencies

For modules that load large models or connect to external services, use unittest.mock:

```python
from unittest.mock import MagicMock, patch
import pytest

def test_with_mocked_model():
    """Test inference logic without loading the actual model."""
    mock_model = MagicMock()
    mock_model.predict.return_value = {"output": "result"}
    
    with patch("api.pneuma.inference.load_model", return_value=mock_model):
        # Your test code
        result = process_with_model()
        assert result is not None
```

### 2. Temporary Directories for File Operations

Use pytest's `tmp_path` fixture for tests that write files:

```python
def test_checkpoint_save_and_load(tmp_path):
    """Test saving and loading model checkpoints."""
    checkpoint_path = tmp_path / "checkpoint.pt"
    
    # Save checkpoint
    save_checkpoint(checkpoint_path)
    assert checkpoint_path.exists()
    
    # Load and verify
    loaded = load_checkpoint(checkpoint_path)
    assert loaded is not None
```

### 3. Testing Resource Management

For tests involving GPU, CPU, or memory resource allocation:

```python
@pytest.mark.slow
def test_resource_scheduling():
    """Test device selection and memory management."""
    scheduler = Scheduler()
    
    # Mock device availability
    with patch("api.gpu.get_available_devices") as mock_devices:
        mock_devices.return_value = [Device.CUDA, Device.CPU]
        device = scheduler.select_device(required_memory=8192)
        assert device in [Device.CUDA, Device.CPU]
```

### 4. Async Tests

For async functions, use pytest-asyncio:

```python
import pytest

@pytest.mark.asyncio
async def test_async_model_load():
    """Test asynchronous model loading."""
    model = await load_model_async("path/to/model")
    assert model is not None
```

### 5. Parametrized Tests

Test multiple inputs with one test function:

```python
import pytest

@pytest.mark.parametrize("input_value,expected", [
    (1, 1),
    (5, 120),  # factorial
    (0, 1),
])
def test_factorial(input_value, expected):
    assert factorial(input_value) == expected
```

## High-Priority Uncovered Modules

These modules are primary candidates for new tests:

1. **`api/resource.py`** (41KB)
   - Resource scheduling and allocation
   - Device management
   - Memory profiling and optimization
   - **Focus**: Scheduler class, device selection logic

2. **`api/gpu.py`**
   - GPU availability detection
   - CUDA initialization
   - **Focus**: Device enum and utilities

3. **`api/model_progress.py`**
   - Model download progress tracking
   - **Focus**: Progress calculation and updates

4. **`api/pneuma/inference/`** (multiple modules)
   - F0 pitch detection pipeline (`f0/`)
   - Voice conversion and TTS inference
   - Model device routing
   - **Focus**: Pure tensor operations, config validation

5. **`api/tts/`**
   - Text-to-speech pipeline modules
   - **Focus**: Audio processing utilities

6. **`api/stt/`**
   - Speech-to-text components

## Coverage Configuration

Coverage is configured in `pyproject.toml` with:
- **Minimum branch coverage**: Enforced at CI time
- **Excluded lines**: Standard pragmas (abstract methods, test code)
- **Source paths**: Only `api/` directory is counted

## CI Integration

Coverage reports are collected in CI and can be:
- Tracked over time for regression detection
- Enforced with minimum thresholds (when CI workflow is added)
- Published as artifacts for review

## Adding New Tests

When adding tests for an uncovered module:

1. Create `tests/test_module_name.py` following existing patterns
2. Use appropriate markers (`@pytest.mark.unit`, etc.)
3. Mock expensive dependencies (model loading, network calls)
4. Use `tmp_path` fixture for file I/O
5. Include docstrings explaining what behavior is tested
6. Run locally: `pytest tests/test_module_name.py -v --cov=api.module_name`
7. Ensure coverage for the module is above 80% if possible

## Troubleshooting

### Tests fail with "module not found"
- Ensure `server/src` is in PYTHONPATH: `export PYTHONPATH=server/src:$PYTHONPATH`
- Or run tests from `server/` directory

### Mocking doesn't work as expected
- Use `patch()` with full module path: `patch("api.module.function")`
- Check that the patch path matches where the function is *used*, not where it's *defined*

### Slow tests timeout
- Mark with `@pytest.mark.slow` and skip with `-m "not slow"`
- Increase timeout with `@pytest.mark.timeout(60)`

### Coverage report shows 0%
- Check that `[tool.coverage.run]` source points to correct directory
- Verify tests are actually exercising the code being measured

## See Also

- pytest documentation: https://docs.pytest.org/
- unittest.mock reference: https://docs.python.org/3/library/unittest.mock.html
- coverage.py: https://coverage.readthedocs.io/
