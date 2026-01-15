.PHONY: docs-html
docs-html:
	pip install -e ".[docs]" && sphinx-build docs/source/ docs/build/

.PHONY: test
test:
	pytest .

.PHONY: lint
lint:
	pylint .
