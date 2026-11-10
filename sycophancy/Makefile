# local quality assurance checks
all: sort format lint typecheck

check: sort-check format-check lint-check typecheck-check

sort:
	isort src

sort-check:
	isort -c src

format:
	black src

format-check:
	black --check src

lint:
	flake8 src

lint-check: lint

typecheck:
	mypy src

typecheck-check: typecheck

test:
	pytest src
