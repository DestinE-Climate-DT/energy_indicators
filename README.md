# Energy Indicators (formerly Energy Onshore)

![latest_release](https://earth.bsc.es/gitlab/digital-twins/de_340-2/energy_onshore/-/badges/release.svg)

This repository contains the scripts related to the Energy Indicators application, part of the Energy use case of the Climate Change Adaptation Digital Twin (Climate DT). All the work is being developed in the frame of the [Destination Earth initiative](https://destination-earth.eu/) from the European Commission, where [ECMWF](https://destine.ecmwf.int/) is one of the Entrusted Entities.

LICENSE NOTE: the European Union, represented by the European Commission is the direct and sole owner of the intellectual property rights of these Results. 

## Description

The Energy Indicators application is currently being developed as a Python package, with two core scripts, `wind.py`, containing a comprehensive set of wind energy indicators and `solar.py`, containing a limited set of solar energy indicators, supporting scripts containing auxiliary functions for data pre- and post-processing, `core.py`, and a wrapper script to envelope the whole structure, `run_energy_onshore.py`.

## Implemented indicators

### Wind energy indicators:

- **Wind Direction** \
    Wind direction from u and v components.
- **Wind Speed Anomalies** \
    Wind anomalies compared to a 30-year baseline reference period.
- **Wind Power Density (WPD)** \
    Amount of energy available in wind at a given location.
- **Capacity Factor (CF)** \
    Ratio of actual energy produced by a wind turbine compared to its maximum theoretical output.
- **Capacity Factor Histogram** \
    Histogram of capacity factors over a 2D grid.
- **Capacity Factor Histogram (1D)** \
    Histogram of capacity factors at a given location.
- **Wind Speed Histogram** \
    Histogram of wind speed over a 2D grid. Deprecated function, now produced by the one-pass layer.
- **Wind Speed Histogram (1D)** \
    Histogram of wind speed at a given location. Deprecated function, now produced by the one-pass layer.
- **Annual Energy Production (AEP)** \
    Energy produced by a wind turbine / wind farm over a year.
- **High Wind Events** \
    Number of times wind speed exceeds a given threshold.
- **Low Wind Events** \
    Number of times wind speed is below a given threshold.
- **Calm Days** \
    Number of days with wind speed below a given threshold.
- **Windy Days** \
    Number of days with wind speed above a given threshold.
- **Cooling Degree Days (CDD)** \
    Weather-based index designed to describe the energy requirements of buildings in terms of cooling.
- **Heating Degree Days (HDD)** \
    Weather-based index designed to describe the energy requirements of buildings in terms of heating.


<Details>

- **Wind direction**: `wind_direction(u, v, mask=None)` \
Compute wind direction from u and v components.
    - Input:
        - `u: xarray.DataArray ; (time,lat,lon)` -> U-component of wind.
        - `v: xarray.DataArray ; (time,lat,lon)` -> V-component of wind.
        - `mask: str or None` -> Path to a netCDF file containing a land-sea mask. If provided, the mask will be applied.

    - Output:
        - `wd: xarray.DataArray ; (time,lat,lon)` -> Wind direction in degrees.

    - References \
        [1]: 

- **Wind speed anomalies**: `wind_speed_anomalies(ws, climatology, scale='daily')` \
Compute the wind speed anomalies.
    - Input:
        - `ws: xarray.DataArray ; (time,lat,lon)` -> Wind speed magnitude.
        - `climatology: xarray.DataArray ; (time,lat,lon)` -> Climatology of wind speed over a 30-year reference period.
        - `scale: str` -> Temporal scale of the climatology. Options are 'daily', 'monthly' and 'yearly'. (default: 'daily')

    - Output:
        - `ws_anom: xarray.DataArray ; (time,lat,lon)` -> Wind speed anomalies.

    - References \
        [1]: 

- **Wind power density**: `wind_power_density(ws, air_density=1.225, mask=None)` \
Compute the wind power density.
    - Input:
        - `ws: xarray.DataArray ; (time,lat,lon)` -> Wind speed magnitude.
        - `air_density: float` -> Air density. Default value is 1.225 kg m^(-3).
        - `mask: str or None` -> Path to a netCDF file containing a land-sea mask. If provided, the mask will be applied.

    - Output:
        - `wpd: xarray.DataArray ; (time,lat,lon)` -> Wind power density.

    - References \
        [1]: 

- **Capacity factor**: `capacity_factor(ws, iec_class, mask=None)` \
Compute the capacity factor of a wind turbine.
    - Input:
        - `ws: xarray.DataArray ; (time,lat,lon)` -> Wind speed magnitude at hub height.
        - `iec_class: str` -> IEC wind turbine class. Options are 'I','I/II', 'II', 'II/III', 'III', 'S'.
        - `mask: str or None` -> Path to a netCDF file containing a land-sea mask. If provided, the mask will be applied.

    - Output:
        - `cf: xarray.DataArray ; (time,lat,lon)` -> Capacity factor.

    - References \
        [1]: https://doi.org/10.1016/j.renene.2019.04.135

- **Capacity factor histogram**: `capacity_factor_histogram(ws, bins, iec_class)` \
Compute the capacity factor histogram of a wind turbine over a 2D grid. 
    - Input:
        - `ws: xarray.DataArray ; (time,lat,lon)` -> Wind speed magnitude at hub height.
        - `bins: int` -> Number of bins.
        - `iec_class: str` -> IEC wind turbine class. Options are 'I','I/II', 'II', 'II/III', 'III', 'S'.
    
    - Output:
        - `counts: xarray.DataArray ; (lat,lon)` -> Number of counts in each bin.
        - `bin_edges: xarray.DataArray ; (lat,lon)` -> Bin edges.

    - References \
        [1]: https://doi.org/10.1016/j.renene.2019.04.135

- **Capacity factor histogram (1D)**: `capacity_factor_histogram_1D(ws, bins, target_lon, target_lat, iec_class)` \
Compute the capacity factor histogram of a wind turbine at a given location.
    - Input:
        - `ws: xarray.DataArray ; (time,lat,lon)` -> Wind speed magnitude at hub height.
        - `bins: int` -> Number of bins.
        - `target_lon: float` -> Longitude of the target location.
        - `target_lat: float` -> Latitude of the target location.
        - `iec_class: str` -> IEC wind turbine class. Options are 'I','I/II', 'II', 'II/III', 'III', 'S'.

    - Output:
        - counts: xarray.DataArray ; (bins) -> Number of counts in each bin.
        - bin_edges: xarray.DataArray ; (bins+1) -> Bin edges.

    - References \
        [1]: https://doi.org/10.1016/j.renene.2019.04.135

- **Wind speed histogram**: `wind_speed_histogram(ws, bins)` \
Compute the wind speed histogram over a 2D grid.
    - Input:
        - `ws: xarray.DataArray ; (time,lat,lon)` -> Wind speed magnitude at hub height.
        - `bins: int` -> Number of bins.
    
    - Output:
        - `counts: xarray.DataArray ; (lat,lon)` -> Number of counts in each bin.
        - `bin_edges: xarray.DataArray ; (lat,lon)` -> Bin edges.

    - References \
        [1]: https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html \
        [2]: https://numpy.org/doc/stable/reference/generated/numpy.histogram.html

- **Wind speed histogram (1D)**: `wind_speed_histogram_1D(ws, bins, target_lon, target_lat)` \
Compute the wind speed histogram at a given location.
    - Input:
        - `ws: xarray.DataArray ; (time,lat,lon)` -> Wind speed magnitude at hub height.
        - `bins: int` -> Number of bins.
        - `target_lon: float` -> Longitude of the target location.
        - `target_lat: float` -> Latitude of the target location.
    
    - Output:
        - `counts: xarray.DataArray ; (bins) -> Number of counts in each bin.
        - `bin_edges: xarray.DataArray ; (bins+1) -> Bin edges.

    - References \
        [1]: https://numpy.org/doc/stable/reference/generated/numpy.histogram.html

- **Annual energy production**: `annual_energy_production_wind(capacity_factor, rated_power, num_turbines=1)` \
Compute the annual energy production of a wind turbine from its capacity factor time series.
    - Input:
        - `capacity_factor: xarray.DataArray ; (time)` -> Capacity factor.
        - `rated_power: float / int` -> Rated power of the wind turbine in MW.
        - `num_turbines: int` -> Number of wind turbines. (default: 1)

    - Output:
        - `aep: xarray.DataArray ; (time)` -> Annual energy production in MWh.

    - References \
        [1]: https://doi.org/10.1016/j.renene.2019.04.135

- **High wind events**: `high_wind_events(ws, threshold=25.0, mask=None)` \
Compute where and when wind speed exceeds a given threshold (cut-out speed).
    - Input:
        - `ws: xarray.DataArray ; (time,lat,lon)` -> Wind speed magnitude at hub height.
        - `threshold: float` -> Wind speed threshold (default: 25.0 m/s). Cut-out speed of the wind turbine.
        - `mask: str or None` -> Path to a netCDF file containing a land-sea mask. If provided, the mask will be applied.

    - Output:
        - `hwe: xarray.DataArray ; (lat,lon)` -> Number of high wind events.

    - References \
        [1]: https://iopscience.iop.org/article/10.1088/1748-9326/acbdb2

- **Low wind events**: `low_wind_events(ws, threshold=3.0, mask=None)` \
Compute where and when wind speed is below a given threshold (cut-in speed).
    - Input:
        - `ws: xarray.DataArray ; (time,lat,lon)` -> Wind speed magnitude at hub height.
        - `threshold: float` -> Wind speed threshold (default: 3.0 m/s). Cut-in speed of the wind turbine.
        - `mask: str or None` -> Path to a netCDF file containing a land-sea mask. If provided, the mask will be applied.

    - Output:
        - `lwe: xarray.DataArray ; (lat,lon)` -> Number of low wind events.

    - References \
        [1]: https://iopscience.iop.org/article/10.1088/1748-9326/acbdb2

- **Cooling degree days**: `cooling_degree_days(tm, tx, tn, base=22.0)` \
Compute the average cooling degree days. Requires daily mean, maximum and minimum temperature.
    - Input:
        - `tm: xarray.DataArray ; (time,lat,lon)` -> Mean temperature.
        - `tx: xarray.DataArray ; (time,lat,lon)` -> Maximum temperature.
        - `tn: xarray.DataArray ; (time,lat,lon)` -> Minimum temperature.
        - `base: float` -> Base temperature (default: 22.0°C). Depends on the region/country considered.
    
    - Output:
        - `cdd: xarray.DataArray ; (time,lat,lon)` -> Cooling degree days.
        - `cdd_acc: xarray.DataArray ; (lat,lon)` -> Total accumulated cooling degree days.

    - References \
        [1]: https://doi.org/10.1002/joc.3959

- **Heating degree days**: `heating_degree_days(tm, tx, tn, base=15.5)` \
Compute the average heating degree days. Requires daily mean, maximum and minimum temperature.
    - Input:
        - `tm: xarray.DataArray ; (time,lat,lon)` -> Mean temperature.
        - `tx: xarray.DataArray ; (time,lat,lon)` -> Maximum temperature.
        - `tn: xarray.DataArray ; (time,lat,lon)` -> Minimum temperature.
        - `base: float` -> Base temperature (default: 15.5°C). Depends on the region/country considered.
    
    - Output:
        - `hdd: xarray.DataArray ; (time,lat,lon)` -> Heating degree days.
        - `hdd_acc: xarray.DataArray ; (lat,lon)` -> Total accumulated heating degree days.

    - References \
        [1]: https://doi.org/10.1002/joc.3959

- **Calm days**: `calm_days(ws, threshold=2.0, mask=None)` \
Compute where and when daily average wind speed is below a given threshold (calm days).
    - Input:
        - `ws: xarray.DataArray ; (time,lat,lon)` -> Daily wind speed magnitude.
        - `threshold: float` -> Wind speed threshold (default: 2.0 m/s).
        - `mask: str or None` -> Path to a netCDF file containing a land-sea mask. If provided, the mask will be applied.

    - Output:
        - `calm: xarray.DataArray ; (lat,lon)` -> Number of calm days.

    - References \
        [1]: https://xclim.readthedocs.io/en/stable/indices.html#xclim.indices.calm_days

- **Windy days**: `windy_days(ws, threshold=10.8, mask=None)` \
Compute where and when daily average wind speed is above a given threshold (windy days).
    - Input:
        - `ws: xarray.DataArray ; (time,lat,lon)` -> Daily wind speed magnitude.
        - `threshold: float` -> Wind speed threshold (default: 10.8 m/s).
        - `mask: str or None` -> Path to a netCDF file containing a land-sea mask. If provided, the mask will be applied.

    - Output:
        - `windy: xarray.DataArray ; (lat,lon)` -> Number of windy days.

    - References \
        [1]: https://xclim.readthedocs.io/en/stable/indices.html#xclim.indices.windy_days

</Details>

### Solar energy indicators:

- **Solar Capacity Factor (daily)** \
    Capacity factor of a PV solar panel at daily scale.
- **Annual Energy Production (daily)** \
    Annual energy production of a PV solar panel at daily scale.

<Details>

- **Solar Capacity Factor (daily)**: `solar_capacity_factor_daily(t2c, rsds)` \
Compute the capacity factor of a PV solar panel at daily scale.
    - Input:
        - `t2c: xarray.DataArray ; (time,lat,lon)` -> Daily temperature at 2m in °C.
        - `rsds: xarray.DataArray ; (time,lat,lon)` -> Daily surface solar radiation donwnwards in W m^(-2).
    
    - Output:
        - `cf_daily: xarray.DataArray ; (time,lat,lon)` -> Daily capacity factor.

    - References \
        [1]: https://doi.org/10.1016/j.renene.2015.10.006 \ 
        [2]: https://doi.org/10.1038/ncomms10014

- **Annual Energy Production (daily)**: `annual_energy_production_daily(capacity_factor, rated_power, num_panels=1)` \
Compute the annual energy production of a PV solar panel.
    - Input:
        - `capacity_factor: xarray.DataArray ; (time)` -> Capacity factor time series for a year.
        - `rated_power: float / int` -> Rated power of the solar panel in kW.
        - `num_panels: int` -> Number of solar panels. (default: 1)

    - Output:
        - `aep_daily: xarray.DataArray ; (time)` -> Annual energy production in kWh.

</Details>

## Version
Current version can be found at the latest publised tags in the git information.

## How to run

Each function/indicator includes a description of its aim, inputs, outputs and corresponding references. The following is an example of how to run the `capacity_factor` function in a Jupyter Notebook / Python environment:

```
import xarray as xr
from energy_onshore.core import wind_speed
from energy_onshore import capacity_factor

# Load wind speed data
path_to_data = 'path/to/data/'
data = xr.open_dataset(path_to_data + 'data.nc')
data.close()

u100 = data['u']
v100 = data['v']

# Compute wind speed
ws = wind_speed(u100, v100)

# Compute capacity factor
cf = capacity_factor(ws, iec_class='I')

# Save capacity factor to netCDF
path_to_output = 'path/to/output/'
cf.to_netcdf(path_to_output + 'capacity_factor.nc')
```

The same procedure can be followed for the rest of the indicators, adjusting the input data and parameters accordingly.

## Input & Processing

The different indicators are built to work with data array objects from the `xarray` library: `xarray.DataArray`

Several support functions have been implemented to facilitate the pre-processing of input data. These functions are included in `core.py`

<Details>

- `check_temperature(data)`
Check if temperature is in Kelvin or Celsius. 
    - Input:
        - `data: xarray.Dataset / xarray.DataArray` -> Temperature data.
    
    - Output:
        - `unit: str` -> Temperature unit of the data. Possible values are 'C' (Celsius) and 'K' (Kelvin).

- `convert_temperature(t, unit='C')`
Convert temperature from Kelvin to Celsius or from Celsius to Kelvin.
    - Input:
        - `t: xarray.DataArray ; (time,lat,lon)` -> Temperature.
        - `unit: str` -> Unit to convert to (default: 'C').
    
    - Output:
        - `t_conv: xarray.DataArray ; (time,lat,lon)` -> Converted temperature.

- `wind_speed(u, v)`
Compute wind speed magnitude from u and v components.
    - Input:
        - `u: xarray.DataArray ; (time,lat,lon)` -> U-component of wind.
        - `v: xarray.DataArray ; (time,lat,lon)` -> V-component of wind.
    
    - Output:
        - `ws: xarray.DataArray ; (time,lat,lon)` -> Wind speed magnitude.

- `cosine_sza_hourly(start_date, end_date, lats, lons)`
Computes the cosine of the Solar Zenith Angle (SZA).
    - Input:
        - `start_date: numpy.datetime64` -> Start date for the time period of interest.
        - `end_date: numpy.datetime64` -> End date for the time period of interest.
        - `lats: numpy.ndarray` -> Latitude values.
        - `lons: numpy.ndarray` -> Longitude values.
    
    - Output:
        - `cossza: xarray.DataArray ; (time,lat,lon)` -> Cosine of the Solar Zenith Angle.

- `percentile(var, percentile, axis=0)`
Compute percentile of a variable along a given axis (i.e. dimension).
    - Input:
        - `var: xarray.DataArray ; (time,lat,lon)` -> Variable.
        - `percentile: float` -> Percentile to compute (0 to 100).
        - `axis: int` -> Axis along which to compute the percentile. Default is 0 (time dimension).
    
    - Output:
        - `perc: xarray.DataArray ; (lat,lon)` -> Percentile of the variable.

- `moving_average(data, window_size)`
Compute the moving average of a variable for the previous 'window_size' time steps.
    - Input:
        - `data: xarray.DataArray ; (time,lat,lon)` -> Variable.
        - `window_size: int` -> Window size of the moving average.

    - Output:
        - `avg: xarray.DataArray ; (time,lat,lon)` -> Moving average of the variable.

- `temporal_rescaling(var, scale='None')`
Rescale a variable to a different temporal frequency.
    - Input:
        - `var: xarray.DataArray ; (time,lat,lon)` -> Variable.
        - `scale: str` -> Temporal frequency to which the variable is rescaled. Possible values are
        'daily', 'weekly', 'monthly', 'annual', and 'seasonal'. Default is 'None'.
    
    - Output:
        - `out: xarray.DataArray ; (time,lat,lon)` -> Rescaled variable.

- `select_point(data, target_lon, target_lat)`
Select the closest point from a data array based on a longitude and latitude of interest.
    - Input:
        - `data: xarray.DataArray ; (time,lat,lon)` -> Data array from which the point is selected.
        - `target_lon: float` -> Longitude of interest.
        - `target_lat: float` -> Latitude of interest.
    
    - Output:
        - `out: xarray.DataArray ; (time)` -> Data array at closest point.

- `select_region(data, latbox, lonbox)`
Select a rectangular region from a DataArray based on a longitude and latitude box.
    - Input:
        - `data: xarray.DataArray ; (time,lat,lon)` -> Data array from which the region is selected.
        - `lonbox: tuple / list` -> Longitude box (min, max).
        - `latbox: tuple / list` -> Latitude box (min, max).
    
    - Output:
        - `out: xarray.DataArray ; (time,lat,lon)` -> Data array at selected region.

- `create_dataset(vars, attrs, coords, dims)`
Creates an xarray dataset with the specified variables, coordinates, dimensions and attributes. Inputs are provided as dictionaries.
    - Input:
        - `vars: dict` -> A dictionary where keys are the variable names and values are numpy arrays or lists.
        - `attrs: dict` -> A dictionary where keys are the variable names and values are numpy arrays or lists. 
        - `coords: dict` -> A dictionary where keys are the coordinate names and values are numpy arrays or lists.
        - `dims: tuple` -> A tuple of dimension names in the order they should appear in the dataset.
    
    - Output:
        - `ds: xarray.Dataset` -> An xarray Dataset containing the specified variables, attributes, coordinates and dimensions.

- `get_type(x)`
Return the type of the variable in string format. Checks if variable is of the correct type.
    - Input:
        - `x: any` -> Variable to check. 'x' can be of any type.
    
    - Output:
        - `type: str` -> Variable type in string format.

- `check_dims(data, n_dims)`
Check if the number of dimensions of a xarray data structure is correct.
    - Input:
        - `data: xarray.Dataset / xarray.DataArray / list of xarray.Dataset / list of xarray.DataArray` -> Data structure to check.
        - `n_dims: int` -> Number of dimensions. Default is 3.

    - Output:
    If the number of dimensions is correct, return None. Otherwise, raise an error.

</Details>

## Output & Visualization

The output of the functions follows the same `xarray.DataArray` structure and can be directly stored in netCDF format. As of now, the package does not include a visualization module, but it is planned to be implemented in future versions.

## Roadmap

Next versions will include a visualization module to plot the results of the different indicators.

## Dependencies

See `setup.py`.

## Support

For any feedback, comments and/or issues you can contact me through Gitlab or directly by email at aleksander.lacima@bsc.es

-------
To install the necessary dependencies for the package:
```
pip install git+https://earth.bsc.es/gitlab/digital-twins/de_340/energy_onshore.git@main
```

To copy the repository to your local directory:
```
git clone https://earth.bsc.es/gitlab/digital-twins/de_340/energy_onshore.git
```

To install the package locally from the root directory (where the `setup.py` file is located):
```
pip install .
```

To check the version of the package in Python:
```
>>> import energy_onshore
>>> energy_onshore.__version__
```

If you are installing the demonstrator version from Github:

```
pip install git+https://github.com/DestinE-Climate-DT/energy_onshore_demonstrator.git@main

git clone https://github.com/DestinE-Climate-DT/energy_onshore_demonstrator.git
```

## How to test:

1. Clone the repository 

```
git clone https://earth.bsc.es/gitlab/digital-twins/de_340-2/energy_onshore.git
```

2. Create a virtual environment

```
python -m venv venv
```

3. activate environment

```
source venv/bin/activate
```

4. pip install in edit mode `-e` and with the testing dependencies `[test]`

```
pip install -e .[test]
```

5. Run pytest

```
pytest .
```

6. Advanced: use pytest to debug tests.

```
pytest --pdb
```

## How to crete the stac catalog from the output:

### Non-DestinE catalog:

1. create a `data` directory, where you store your output.

```
mkdir data
```

2. run the `generate_stac_catalog.py` sctipt (one dir above `data`).

```
python3 generate_stac_catalog.py
```

### How to access the documentation:

You can either build and serve the documentation locally:

```
pip install -e ".[docs]" && sphinx-build docs/source/ docs/build/ && firefox docs/build/index.html
```

or look at the github pages (may have some delay).

https://destine-climate-dt.github.io/energy_indicators/

### Catalog to be transferred to the Eumetsat Datalake:

Follow these instructions https://destine-data-lake-docs.data.destination-earth.eu/en/latest/dedl-discovery-and-data-access/User-Generated-Data/Promote-user-data-to-become-DestinE-data/Promote-user-data-to-become-DestinE-data.html#step-4-data-preparation

### License

Copyright 2022-2025 European Union (represented by the European Commission)
The Energy Indicators package is distributed as open-source software under Apache 2.0 License. The copyright owner is the European Union, represented by the European Commission. The development of the Energy Indicators package has been funded by the European Union through Contract DE_340_CSC - Destination Earth Programme Climate Adaptation Digital Twin (Climate DT). Further info can be found at https://destine.ecmwf.int/ and https://destination-earth.eu/
