SHELL := /bin/bash

set-dev-env:
	python3 -m venv dev_env && \
	source dev_env/bin/activate && \
	pip install -r requirements-dev.txt

activate-dev:
	source dev_env/bin/activate
