# Changelog


## v0.7.4
# Major (major changes): 

# Minor (new features):
- unit tests are now extended to `run_energy_onshore.py` by using reduced (10x10) test data coming from real data from gsv and OPA.
- added a pyproject.toml to solve cicd issues.
- added ploting functions and its tests.
- added a script to convert the output into stac format to transfer to the datalake eventually.

# Patches:

## v0.7.3:
# Major (major changes): 

# Minor (new features):

# Patches:
- froze most dependency versions.
- get the version of the application automatically set at pip install step.

## v0.7.2

# Major (major changes): 

# Minor (new features):

# Patches:
- Fixed unit tests in the CI/CD pipeline.

## v0.7.1 (25/10/2024)

### Major cahnges

### Minor changes
* added placeholders for further unit tests
* Added changelog info for v0.7.0

### bugfixes
* refined linting

## v0.7.0

### Major changes
* restructured transcript so that all the functions are defined inside the package and not inside the workflow repository.

### Minor changes
* Added metadata coming from OPA

### Bugfixes 
* linting

## v0.6.0 (05-07-2024)

### New features

- Added pytests for most of core.py and wind.py. These include the addicion of fixtures and all the tests under /tests

### Format fixes

- Pylint has been applied to all the code to comply to coding standards.
- It has successfully passed and included in a CI/CD pipeline.

## v0.5.0 (13-03-2024)

### New features

- Added a new module for solar indicators:
    - `solar.py`
- Added a new function to compute the solar capacity factor at daily scale:
    - `solar_capacity_factor_daily`
- Added a new function to compute the annual energy production (AEP) of a solar farm based on the capacity factor time series for a whole year:
    - `annual_energy_production_daily`
- Generated new daily climatologies for ws from 1991 to 2020 based on a 3-day running window.
- Improved `wind_speed` and `wind_direction` functions.

### Bug fixes

- Deleted unnecessary files from `power_curves` directory.
- Fixed an issue with leap years in the `wind_speed_anomalies` function.
- Reverted the use of `pathlib` in `power_curve` function as it was not working properly.

## v0.4.0 (28-02-2024)

### New features

- Added a new function to compute the wind speed histogram over a 2D grid (the function is a bit slow if global data is used):
    - `wind_speed_histogram`
- Added a new function to compute the wind speed histogram at a given location:
    - `wind_speed_histogram_1D`
- Added a new function to compute the Annual Energy Production (AEP) of a wind turbine / farm based on the capacity factor time series for a whole year:
    - `annual_energy_production_wind`
- Added a new function to compute the number of high wind speed events above a given threshold:
    - `high_wind_events`
- Added a new function to compute the number of low wind speed events below a given threshold:
    - `low_wind_events`
- Added `cosine_sza_hourly` function to `core.py`, which computes the cosine of the solar zenith angle for each hour of a given time period.
- Generalized the `wind_speed_anomalies` function to work for any region instead of only for the whole globe.
- Added real hub heights to turbine properties in the `power_curve` function.
- Added `pathlib` to the dependencies in `setup.py` as it is now used by the `power_curve` function.

### Bug fixes

- General improvements of the code to increase readability and maintainability.
- Changed `cKDTree` to `KDTree` in the `capacity_factor` function to avoid problems with the former.


## v0.3.0 (09-11-2023)

### New features

- Added a new function to compute the capacity factor histogram over a 2D grid (the function is a bit slow if global data is used):
    - `capacity_factor_histogram`
- Added a new function to compute wind speed anomalies:
    - `wind_speed_anomalies`
- Generated new daily climatologies for wind speed from 1991 to 2020. Adjusted the function `wind_speed_anomalies` to work with the new climatologies.
- Added `select_region` function to `core.py`, which selects a rectangular region from a DataArray based on the limits of the selected domain.
- Renamed `capacity_factor_histogram`, which was designed to compute the histogram at a given location, to `capacity_factor_histogram_1D` to avoid confusion with the new function.
- Removed previous change to `__init__.py` as it was causing problems with the installation.

### Bug fixes

- Fixed a bug in the `create_dataset` function that was causing the datasets to be full of NaNs.

## v0.2.3 (26-10-2023)

### New features

- Added a new function to compute the capacity factor histogram at a given location:
    - `capacity_factor_histogram`
- Modified dimension checks to increase speed.
- Modified the indicators to work with wind speed as an input instead of u and v components.
- Modified `__init__.py` to not import all functions from `wind.py` by default.

### Bug fixes

- Fixed a bug in the `heating_degree_days` function that caused the output to be incorrect.

## v0.2.2 (25-10-2023)

### New features

- Implemented dimension checks in each indicator to ensure that input data has the correct shape.
- Added `check_dims` function to `core.py`, which checks the dimensions of the input data.
- Added `check_temperature` function to `core.py`, which checks the units of the temperature data.
- Removed `relative_humidity` and `vapour_pressure` functions from `core.py` as they are not used in any indicator.
- Modified `setup.py` to read package version from `__init__.py` dinamically.

### Bug fixes

- Fixed `__init__.py` for pip installation to work properly.
- Fixed some minor bugs across several scripts in the library.

## v0.2.1 (03-10-2023)

### New features

- Modified the `capacity_factor` function to work with `cKDTree` instead of `multiprocessing`. The code is now about 90 times faster (each time step is now computed in 20 s instead of 30 min).
- Implemented attributes as input in the `create_dataset` function from `core.py`.


## v0.2.0 (29-09-2023)

### New features

- Added new indicators:
    - Capacity factor
- Added a function to compute the power curve of a wind turbine based on data from the manufacturer and a Weibull CDF.
- Deployed a vectorized implementation of `heating_degree_days` and `cooling_degree_days`. 
- Renamed `processing.py` to `core.py`.
- Added `setup.py` and `__init__.py` files to deploy the application as a package.

### Bug fixes

- Fixed some minor bugs in `wind.py` and `processing.py`.


## v0.1.0 (04-07-2023)

### New features

- First release.
