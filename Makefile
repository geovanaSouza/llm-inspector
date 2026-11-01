# VENV: virtual environment path (default: .venv)
VENV ?= .venv
# PYTHON: Python interpreter used by setup (default: $(VENV)/bin/python)
PYTHON ?= $(VENV)/bin/python
# Default prompt if none is provided via command line
PROMPT ?= "Prompt Telemetry Engine - An LLM mechanic inspector"

.PHONY: setup run clean

# Create a virtual environment and install packages
setup:
	uv venv $(VENV) --python 3.12
	uv pip install --python $(PYTHON) -e .
	@echo "\n✅ Environment configured with pyproject.toml with success!"

# Execute the application
run:
	$(PYTHON) main.py $(PROMPT)

# Clean temporary files, build artifacts and the virtual environment
clean:
	rm -rf venv __pycache__ *.egg-info build
	@echo "🧹 Clean finished!"
