<h1 align="center">
  <img src="https://img.icons8.com/?size=100&id=118557&format=png&color=000000" width="72" style="vertical-align: middle;"/> Testing DeepGit
</h1>

DeepGit includes a comprehensive suite of tests to ensure the reliability of each module.

## Test Suite Overview

Tests are organized under the `tests/` directory, covering the following modules:

- **Query Conversion:** `tests/test_convert_query.py`
- **Repository Ingestion:** `tests/test_github.py`
- **Neural Dense Retrieval:** `tests/test_dense_retrieval.py`
- **Cross-Encoder Re-Ranking:** `tests/test_cross_encoder_reranking.py`
- **Threshold Filtering:** `tests/test_filtering.py`
- **Repository Activity Analysis:** `tests/test_activity_analysis.py`
- **Decision Maker:** `tests/test_decision_maker.py`
- **Code Quality Analysis:** `tests/test_code_quality.py`
- **Merge Analysis:** `tests/test_merge_analysis.py`
- **Multi-Factor Ranking:** `tests/test_ranking.py`
- **Output Presentation:** `tests/test_output_presentation.py`

## Running the Tests

To run the entire test suite, use one of the following commands from the project root:

```bash
pytest
```

Or, if you have a test runner script (e.g., run_tests.py):

```bash
python run_tests.py
```

## Test Environment
**Mocking:**
External dependencies such as HTTP requests (to GitHub) and model predictions are mocked using monkeypatch and dummy functions to ensure tests are deterministic.

**Coverage:**
Each test file simulates various scenarios (e.g., valid data, error handling) to cover the full functionality of each module.

**Troubleshooting**
Ensure your project structure includes all necessary __init__.py files.

Check that any required dummy data or configuration is correctly set in the test files.

Use verbose mode (pytest -v) for more detailed output if tests fail.

