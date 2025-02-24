ENV_NAME=fetch_env

.PHONY: env install test writeup run_all

env:
	python -m venv $(ENV_NAME)


install: env
	$(ENV_NAME)/bin/pip install -r requirements.txt


test: install
	$(ENV_NAME)/bin/python assessment.py


writeup:
	@echo "Write-up can be found in writeup.md"


run_all: env install test writeup

