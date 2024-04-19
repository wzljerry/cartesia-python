install:
	pip install -e .

install-dev:
	pip install -e .
	pip install pytest pytest-cov

autoformat:
	ruff .

lint:
	ruff check .

test:
	pytest -ra tests/ -sv --cov=cartesia/ --log-cli-level=INFO
