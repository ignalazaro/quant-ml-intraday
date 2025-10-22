.PHONY: install format lint typecheck test docs precommit

install:
	poetry install

format:
	poetry run black src tests

lint:
	poetry run ruff check .

typecheck:
	poetry run mypy src

test:
	poetry run pytest --cov=src tests

docs:
	poetry run mkdocs serve

precommit:
	poetry run pre-commit install

report:
	poetry run python -m quantml.pipelines.research
	poetry run python scripts/generate_report.py
	mkdocs serve

