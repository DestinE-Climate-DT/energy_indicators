.PHONY: docs-html
docs-html:
	pip install ".[docs]" && sphinx-build docs/source/ docs/build/
