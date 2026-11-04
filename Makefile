# VENV: virtual environment path (default: .venv)
VENV ?= .venv
# PYTHON: Python interpreter used by setup (default: $(VENV)/bin/python)
PYTHON ?= $(VENV)/bin/python
# Default prompt 1 if none is provided via command line
PROMPT1 ?= "The role of temperature in LLMs is to"
# Default prompt 2 if none is provided via command line
PROMPT2 ?= "How does temperature affect randomness in language models?"

.PHONY: setup run clean

# Create a virtual environment and install packages
setup:
	uv venv $(VENV) --python 3.12
	uv pip install --python $(PYTHON) -e .
	@echo "\n✅ Environment configured with pyproject.toml with success!"

# Execute the application
run:
	$(PYTHON) main.py $(PROMPT1) $(PROMPT2)

# Clean temporary files, build artifacts and the virtual environment
clean:
	rm -rf venv __pycache__ *.egg-info build
	@echo "🧹 Clean finished!"
