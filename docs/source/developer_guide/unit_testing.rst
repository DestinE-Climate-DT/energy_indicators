Unit Testing
============

Running Tests
--------------

.. code-block:: bash

    pytest tests/

or, equivalently:

.. code-block:: bash

    make test

Test Structure
---------------

Each package module has a corresponding test file (``test_core.py``,
``test_wind.py``, ``test_demand.py``, ``test_solar.py``,
``test_mask_processing.py``, ``test_plot.py``), plus
``test_run_energy_onshore.py`` for the ``run_*`` wrapper layer.

Shared synthetic test data (temperature, wind components, radiation, on a
fixed 10×10×7 lat/lon/time grid) is defined once in ``conftest.py`` as
pytest fixtures (``dataarray_u``, ``dataarray_t_c``, etc.) and reused across
test files, rather than each test constructing its own input data. Tests are
expected to contain genuine assertions against this data rather than serve
as empty placeholders.

CI Integration
----------------

The GitLab CI pipeline (``.gitlab-ci.yml``) runs ``pytest .`` on every push,
in a ``test`` stage separate from ``lint`` (pylint, minimum score 8.5) and
``docs`` (Sphinx build). All three must pass before a merge request can be
merged.

Coverage
---------

Coverage is measured via ``pytest-cov`` (configured in ``pytest.ini``) and
printed after each test run. Note: ``pytest.ini`` references a
``.coveragerc`` file for coverage configuration, which is not currently
present in the repository (coverage currently runs with default settings).

