# Makefile — developer entry points for the Intent Classifier repo.
# Run `make help` for the full list. Targets are thin wrappers around the
# existing scripts (see docs/pipelines.md); they do not change any pipeline.

PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
.PHONY: install
install: ## Install exact locked runtime deps (falls back to requirements.txt)
	$(PIP) install -r requirements.lock 2>/dev/null || $(PIP) install -r requirements.txt

.PHONY: lock
lock: ## Regenerate requirements.lock from requirements.txt (needs uv)
	uv pip compile requirements.txt --universal --python-version 3.10 -o requirements.lock

.PHONY: install-dev
install-dev: install ## Install runtime + developer tooling (ruff, black, mypy, pytest, pre-commit)
	$(PIP) install ruff black darker mypy pytest pytest-cov pre-commit \
	  "jsonschema>=4.18" "referencing>=0.30" "cryptography>=42.0" "pyyaml>=6.0"
	pre-commit install

# ---------------------------------------------------------------------------
# Quality gates (mirror CI and pre-commit)
# ---------------------------------------------------------------------------
.PHONY: format
format: ## Format CHANGED lines only (darker vs HEAD) + Ruff autofix
	ruff check --fix .
	darker --revision HEAD .

.PHONY: format-all
format-all: ## One-time full-repo Black pass (deliberate; large diff)
	ruff check --fix .
	black .

.PHONY: lint
lint: ## Lint with Ruff (no changes)
	ruff check .

.PHONY: format-check
format-check: ## Verify changed lines are formatted (darker) + Ruff
	darker --check --revision HEAD .
	ruff check .

.PHONY: typecheck
typecheck: ## Type-check the maintained library code with MyPy
	mypy packages/runtime/nlu_engine packages/buildtime multilingual/*.py

.PHONY: test
test: ## Run the test suite
	pytest

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	pytest --cov --cov-report=term-missing

.PHONY: check
check: lint typecheck test ## Run all quality gates

.PHONY: precommit
precommit: ## Run all pre-commit hooks on all files
	pre-commit run --all-files

# ---------------------------------------------------------------------------
# ML pipelines (see docs/pipelines.md)
# ---------------------------------------------------------------------------
.PHONY: train
train: ## Train the English TF-IDF intent model and export ONNX (version 3 data)
	PYTHONPATH=packages/buildtime $(PYTHON) packages/buildtime/nlu_training/train.py

.PHONY: train-multilingual
train-multilingual: ## Train the multilingual intent model
	$(PYTHON) multilingual/train_multilingual.py

.PHONY: predict
predict: ## Interactive inference CLI (English ONNX model)
	$(PYTHON) apps/cli/predict.py

.PHONY: nlu
nlu: ## Run the full NLU engine CLI
	$(PYTHON) apps/cli/nlu_cli.py

.PHONY: calibrate
calibrate: ## Fit per-language temperature scaling / calibration
	$(PYTHON) packages/buildtime/nlu_training/calibrate_languages.py

# ---------------------------------------------------------------------------
# Model export (mobile: ONNX / CoreML / INT8)
# ---------------------------------------------------------------------------
.PHONY: build-bundle
build-bundle: ## Build + dev-sign a .nlu bundle from an unpacked dir (SRC=..., OUT=...)
	PYTHONPATH=packages/buildtime $(PYTHON) -m nlu_compiler.build $(or $(SRC),spec/examples/3.0/full) $(if $(OUT),--out $(OUT),)

.PHONY: verify-bundle
verify-bundle: ## Verify a .nlu bundle (BUNDLE=...)
	PYTHONPATH=packages/buildtime $(PYTHON) -m nlu_compiler.verify $(or $(BUNDLE),bundles/full.nlu)

.PHONY: export-coreml
export-coreml: ## Export CoreML .mlpackage bundles (FP16 + FP32)
	$(PYTHON) multilingual/export_coreml_multilingual.py --all --fp16 --fp32

.PHONY: export-coreml-test
export-coreml-test: ## Numeric-equivalence (Tier-A) CoreML export test
	$(PYTHON) multilingual/test/test_coreml_multilingual.py --full

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------
.PHONY: clean
clean: ## Remove caches and build artifacts
	find . -type d -name '__pycache__' -not -path './.venv*/*' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache .mypy_cache .pytest_cache .coverage htmlcov coverage.xml build dist

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
