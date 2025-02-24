ENV_NAME=fetch_env

.PHONY: env activate install test writeup run_all

writeup:
	@echo "Write-up can be found in writeup.md"

env:
	python -m venv $(ENV_NAME)


activate:
	source $(ENV_NAME)/bin/activate


install: 
	pip install -r requirements.txt


test:
	python assessment.py


run_all: env activate install test writeup
