New Contributions
==================

Development Setup
-------------------

.. code-block:: bash

    git clone git@gitlab.earth.bsc.es:digital-twins/de_340-3/energy_indicators.git
    cd energy_indicators
    pip install -e ".[all]"

``test`` and ``docs`` extras are also available individually for a lighter
install.

Workflow
---------

NOTE: Do not confuse with the `ClimateDT Workflow <https://gitlab.earth.bsc.es/digital-twins/de_340-3/workflow>`_.

1. Create an issue describing the change, or pick up an existing one.
2. Create a branch from ``main`` (GitLab's "Create branch" button on the
   issue follows the naming convention ``<issue-number>-short-description``).
3. Make the change, with associated unit tests and docstring updates.
4. Commit and push the changes to GitLab, which automatically triggers the
   CI/CD pipeline.
5. Open a merge request into ``main``, referencing the issue
   (``Closes #<issue-number>``).
6. Wait for the CI/CD pipeline to pass (see below) and for review/approval.
7. Merge into ``main`` and update the ``changelog.md``.
8. In some cases, when a set of changes is ready to ship as a new version 
   (e.g. a batch   of features, or a release the team has agreed on) create
   an annotated Git tag (``vX.Y.Z``), and create a GitLab Release from that 
   tag. Tagging and releasing is not done for every merged change. 


CI/CD Pipeline
---------------

Every push runs through four GitLab CI stages, defined in
``.gitlab-ci.yml``:

- **lint**: ``pylint``, minimum score 8.5, run against the package and
  individual core modules.
- **test**: ``pytest .``
- **docs**: builds the Sphinx documentation, to catch build errors early.
- **mirror**: only runs on a tag push; pushes a snapshot to GitHub.

All stages must pass before a merge request can be merged.

Standards
----------

- Code must pass linting (pylint ≥ 8.5).
- New or changed functions require docstrings (parameters, output, and
  references where applicable).
- Changes require corresponding unit tests, see :doc:`unit_testing`.
- Non-trivial changes should be reflected in ``changelog.md`` under the
  current unreleased version entry.

Adding a New Indicator
------------------------

Adding a new physical indicator typically touches three places:

1. **Compute function**: implemented in the relevant module (``wind.py``,
   ``demand.py``, or ``solar.py``), taking ``xarray.DataArray`` input and
   following the existing docstring conventions (Input/Output/References).
2. **Wrapper function**: added to ``run_energy_indicators.py``, following
   the existing ``run_*`` pattern: read the raw input NetCDF file(s), call
   the compute function, attach the global metadata attributes used
   throughout the package, and write the result to an output NetCDF file.
3. **Tests**: both a test in the corresponding ``test_<module>.py``
   exercising the compute function directly, and a test in
   ``test_run_energy_onshore.py`` exercising the ``run_*`` wrapper
   end-to-end (see :doc:`unit_testing`).
4. **Documentation**: add a description of the new indicator to the
   relevant developer/user guide section.

If the indicator also needs to run as part of the operational ClimateDT
workflow, that requires a further step in the separate
`workflow repository <https://gitlab.earth.bsc.es/digital-twins/de_340-2/workflow>`_,
tagging the workflow maintainers to open a branch/merge request there.

