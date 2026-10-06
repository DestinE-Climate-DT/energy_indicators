Package Structure
==================

Repository Layout
------------------

.. code-block:: text

    energy_indicators/          # installable package
    ├── core.py                 # generic array/numerical utilities
    ├── demand.py                # demand indicators
    ├── mask_processing.py       # land-sea mask handling
    ├── plot.py                  # plotting functionality
    ├── power_curves/            # turbine power curves
    ├── run_energy_indicators.py # wrapper functions run by the workflow
    ├── solar.py                 # solar indicators
    ├── utils.py                 # turbine specification lookup
    ├── wind.py                  # wind production and wind statistics indicators
    └── __init__.py

    tests/          # pytest test suite, see Unit Testing
    docs/           # Sphinx documentation source
    run/            # example/demonstration notebooks
    utils/          # maintainer scripts (STAC metadata cataloguing), not
                    # part of the installable package

Core Modules
------------

- **core.py**: generic, dependency-light array/numerical utilities
  (unit conversion, wind speed, percentiles, spatial selection). Imported by
  nearly every other module.
- **wind.py**: wind indicators, including capacity factor, wind power density,
  high/low wind events, calm/windy days, wind direction, OPA-based streaming
  histograms.
- **demand.py**: cooling and heating degree days (CDD/HDD).
- **solar.py**: PV potential; further solar indicators planned.
- **mask_processing.py**: applies a land-sea mask via nearest-neighbour
  interpolation, used across ``wind.py`` and ``demand.py``.
- **utils.py**: static turbine specifications used by ``capacity_factor``.
- **plot.py**: map plotting for indicator output.
- **run_energy_indicators.py**: wrapper layer reading raw input, calling the
  relevant indicator function, attaching metadata, and writing output NetCDF.
  This is what the ClimateDT workflow calls directly.

All of the above are re-exported at the package root
(``energy_indicators/__init__.py``), so functions are typically imported as
``from energy_indicators import run_capacity_factor_i`` rather than via
their individual submodules.

Design Principles
-------------------

- **Modularity**: physical domains (wind, solar, demand) are separated into
  their own modules; new indicators can be added without touching unrelated
  code.
- **Streaming compatibility**: indicators are designed to run concurrently
  with climate simulations via the
  `one_pass <https://gitlab.earth.bsc.es/digital-twins/de_340-2/one_pass>`_
  library, rather than requiring the full time series in memory.
- **Testability**: every module has a corresponding test file, enforced via
  the CI pipeline's ``test`` stage on every push (see :doc:`unit_testing`).
- **Reproducibility**: dependencies are pinned to exact versions in
  ``setup.py``.
- **Separation of computation and orchestration**: indicator modules contain
  pure computation; ``run_energy_indicators.py`` handles I/O and metadata 
  and acts as an interface to the workflow.

