"""
This module contains core functionalities for the onshore energy project.

It includes functions and classes that handle data processing, analysis,
and other core tasks necessary for the project.
"""

# Destination Earth: Energy Onshore application
# Author: Aleks Lacima
# Version: 0.5.0 + dev

import multiprocessing

# Load libraries
import xarray as xr
import numpy as np
import pandas as pd

# GSV data post-processing and basic analysis.


def check_temperature(data):
    """
    Check if temperature is in Kelvin or Celsius.

    Input
    -------
    data: xarray.Dataset / xarray.DataArray
        Temperature data.

    Output
    -------
    unit: str
        Temperature unit of the data.
        Possible values are 'C' (Celsius) and 'K' (Kelvin).
    """
    # Check if the input parameters satisfy the required conditions.
    assert get_type(data) in [
        "Dataset",
        "DataArray",
    ], 'The input variable "data" is not an xarray.Dataset / xarray.DataArray.'

    # Check if temperature is in Kelvin or Celsius.
    try:
        if data.attrs["units"] == "K":
            unit = "K"
        elif data.attrs["units"] == "C":
            unit = "C"
        else:
            raise ValueError("The temperature unit is not valid.")
    except AttributeError:
        raise AttributeError('The data object does not have the attribute "units".')

    return unit


def convert_temperature(t, unit="C"):
    """
    Convert temperature from Kelvin to Celsius or from Celsius to Kelvin.

    Input
    -------
    t: xarray.DataArray ; (time,lat,lon)
        Temperature (in Kelvin/Celsius).
    unit: str
        Temperature unit of the output.
        Possible values are 'C' (Celsius) and 'K' (Kelvin). Default is 'C'.

    Output
    -------
    t_conv: xarray.DataArray ; (time,lat,lon)
        Air temperature at 2m (in Kelvin/Celsius).
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(t) == "DataArray"
    ), 'The input variable "t" is not an xarray.DataArray.'
    unit = unit.upper()
    if unit not in ["C", "K"]:
        raise ValueError("The specified temperature unit is not valid.")

    # Convert from Kelvin to Celsius.
    if unit == "C":
        t_conv = t - 273.15
    # Convert from Celsius to Kelvin.
    elif unit == "K":
        t_conv = t + 273.15

    return t_conv


def wind_speed(u, v):
    """
    Compute wind speed magnitude from u and v components.

    Input
    -------
    u: xarray.DataArray ; (time,lat,lon)
        U-component of wind.
    v: xarray.DataArray ; (time,lat,lon)
        V-component of wind.

    Output
    -------
    ws: xarray.DataArray ; (time,lat,lon)
        Wind speed magnitude.
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(u) == "DataArray"
    ), 'The input variable "u" is not an xarray.DataArray.'
    assert (
        get_type(v) == "DataArray"
    ), 'The input variable "v" is not an xarray.DataArray.'

    # Compute wind speed magnitude.
    ws = np.sqrt(u**2 + v**2)

    # Add metadata to the output variable.
    attrs = {"shortname": "ws", "longname": "Wind speed", "units": "m/s"}
    coords = {"time": u.time, "lat": u.lat, "lon": u.lon}
    dims = ("time", "lat", "lon")

    ws = xr.DataArray(
        ws, dims=dims, coords=coords, attrs=attrs, name=attrs["shortname"]
    )

    return ws


def wind_direction(u, v):
    """
    Compute wind direction from u and v components.

    Input
    -------
    u: xarray.DataArray ; (time,lat,lon)
        U-component of wind.
    v: xarray.DataArray ; (time,lat,lon)
        V-component of wind.

    Output
    -------
    wd: xarray.DataArray ; (time,lat,lon)
        Wind direction.
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(u) == "DataArray"
    ), 'The input variable "u" is not an xarray.DataArray.'
    assert (
        get_type(v) == "DataArray"
    ), 'The input variable "v" is not an xarray.DataArray.'

    # Compute wind direction.
    wd = np.arctan2(u, v)

    # Convert wind direction from radians to degrees.
    wd = np.degrees(wd)

    # Add metadata to the output variable.
    attrs = {"shortname": "wd", "longname": "Wind direction", "units": "degrees"}
    coords = {"time": u.time, "lat": u.lat, "lon": u.lon}
    dims = ("time", "lat", "lon")

    wd = xr.DataArray(
        wd, dims=dims, coords=coords, attrs=attrs, name=attrs["shortname"]
    )

    return wd


def cosine_sza_hourly(start_date, end_date, lats, lons):
    """
    Computes the cosine of the Solar Zenith Angle (SZA) at an hourly frequency with day/night \
    masking. Nighttime values are set to np.nan.

    Input
    -------
    start_date: numpy.datetime64
        Start date for the time period of interest.
    end_date: numpy.datetime64
        End date for the time period of interest.
    lats: numpy.ndarray
        Array of latitudes.
    lons: numpy.ndarray
        Array of longitudes.

    Output
    -------
    cossza: xarray.DataArray ; (time,lat,lon)
        Cosine of the Solar Zenith Angle.

    References
    -------
    [1]: https://doi.org/10.1002/2015GL066868
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(start_date) == "datetime64"
    ), 'The input variable "start_date" is not a numpy.datetime64.'
    assert (
        get_type(end_date) == "datetime64"
    ), 'The input variable "end_date" is not a numpy.datetime64.'
    assert (
        get_type(lats) == "ndarray"
    ), 'The input variable "lats" is not a numpy.ndarray.'
    assert (
        get_type(lons) == "ndarray"
    ), 'The input variable "lons" is not a numpy.ndarray.'

    # Degrees to radians conversion factor
    deg_to_rad = np.pi / 180.0

    # Generate hourly time range
    time_index = pd.date_range(start=start_date, end=end_date, freq="h")

    # Prepare latitude and longitude meshgrid
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Initialize cosine SZA array
    cosine_sza = np.full(
        (len(time_index), len(lats), len(lons)), np.nan
    )  # Initialize with np.nan for night
    # sza= np.full((len(time_index), len(lats), len(lons)), np.nan)

    for i, time in enumerate(time_index):
        # Day of year
        day_of_year = time.day_of_year
        ndays = 365

        # Solar declination (delta)
        declination = 23.45 * np.sin(deg_to_rad * 360.0 / ndays * (day_of_year - 81))

        # Time correction for solar noon
        equation_of_time = (
            9.87 * np.sin(2 * 2 * np.pi / ndays * day_of_year)
            - 7.53 * np.cos(2 * np.pi / ndays * day_of_year)
            - 1.5 * np.sin(2 * np.pi / ndays * day_of_year)
        )

        time_correction = (
            4 * (lon_grid - 0) + equation_of_time
        )  # Assuming GMT time zone; adjust '0' accordingly

        # Solar hour angle (HRA)
        solar_time = (time.hour * 60 + time.minute + time_correction) / 60.0
        hour_angle = 15 * (solar_time - 12)

        # Solar zenith angle (theta) and its cosine
        declination_rad = declination * deg_to_rad
        lat_rad = lat_grid * deg_to_rad
        hour_angle_rad = hour_angle * deg_to_rad
        cos_theta = np.sin(lat_rad) * np.sin(declination_rad) + np.cos(
            lat_rad
        ) * np.cos(declination_rad) * np.cos(hour_angle_rad)
        theta = (
            np.arccos(np.clip(cos_theta, -1, 1)) / deg_to_rad
        )  # Clipping cos_theta for numerical stability

        # Apply day/night mask: Update cosine SZA if it's day (theta < 90 degrees)
        is_day = theta < 90
        # sza_array[i, :, :][is_day] = theta[is_day]
        cosine_sza[i, :, :][is_day] = np.clip(
            cos_theta[is_day], -1, 1
        )  # Ensure the cosine value is within [-1, 1]

    # Add metadata to the output variable.
    attrs = {
        "shortname": "cossza",
        "longname": "Cosine Solar Zenith Angle",
        "units": "-",
    }
    coords = {"time": time_index, "lat": lats, "lon": lons}
    dims = ("time", "lat", "lon")

    cossza = xr.DataArray(
        cosine_sza, dims=dims, coords=coords, attrs=attrs, name=attrs["shortname"]
    )

    return cossza


# Statistical analysis.


def percentile(var, wanted_percentile, axis=0):
    """
    Compute percentile of a variable along a given axis (i.e. dimension).

    Input
    -------
    var: xarray.DataArray ; (time,lat,lon)
        Variable.
    wanted_percentile: float
        Percentile to compute (0 to 100).
    axis: int / tuple
        Axis along which to compute the percentile. Default is 0 (time dimension).

    Output
    -------
    perc: xarray.DataArray ; (lat,lon)
        Percentile of the variable.
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(var) == "DataArray"
    ), 'The input variable "var" is not an xarray.DataArray.'
    assert (
        get_type(wanted_percentile) == "float"
    ), 'The input variable "percentile" is not a float.'
    assert (
        get_type(axis) == "int" or get_type(axis) == "tuple"
    ), 'The input variable "axis" is not an int / tuple.'

    # Compute percentile of the variable.
    perc = np.percentile(var, wanted_percentile, axis=axis)

    return perc


def moving_average(data, window_size):
    """
    Compute the moving average of a variable for the previous 'window_size' time steps.

    Input
    -------
    data: numpy.ndarray ; (time,lat,lon)
        Variable.assert
    Output
    -------
    avg: xarray.DataArray ; (time,lat,lon)
        Moving average of the variable.
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(data) == "ndarray"
    ), 'The input variable "data" is not an numpy.ndarray.'
    assert (
        get_type(window_size) == "int"
    ), 'The input variable "window_size" is not an int.'

    # Compute moving average of the variable.
    avg = np.cumsum(data, axis=0)
    avg[window_size:] = avg[window_size:] - avg[:-window_size]

    return avg[window_size - 1 :] / window_size


def spatial_multiprocessing(function, var, *args):
    """
    Parallelise a calculation over a 2D grid. The calculation is performed in parallel for each \
    grid cell.

    Input
    -------
    function: function
        Function to be parallelised.
    var: xarray.DataArray ; (lat,lon)
        Variable.
    *args: tuple
        Arguments of the function to be parallelised.

    Output
    -------
    res: xarray.DataArray ; (lat,lon)
        Output of the parallelised function.
    """

    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(function) == "function"
    ), 'The input variable "function" is not a function.'
    assert (
        get_type(var) == "DataArray"
    ), 'The input variable "var" is not an xarray.DataArray.'
    assert get_type(args) == "tuple", 'The input variable "*args" is not a tuple.'

    # Create a list of all (i, j) positions
    jobargs = [
        (i, j, var, args) for i in range(var.shape[0]) for j in range(var.shape[1])
    ]

    # Create a pool with the number of desired processes
    num_processes = (
        multiprocessing.cpu_count()
    )  # Adjust this value based on your system capabilities
    with multiprocessing.Pool(num_processes) as pool:
        out = list(pool.starmap(function, jobargs))
    pool.close()
    pool.join()

    res = np.array(out).reshape(var.shape[0], var.shape[1])

    return res


# Data restructuring and resampling.


def temporal_rescaling(var, scale="None"):
    """
    Rescale a variable to a different temporal frequency.

    Input
    -------
    var: xarray.DataArray ; (time,lat,lon)
        Variable to be restructured.
    scale: str
        Temporal frequency to which the variable is rescaled. Possible values are
        'daily', 'weekly', 'monthly', 'annual', and 'seasonal'. Default is 'None'.

    Output
    -------
    out: xarray.DataArray ; (time,lat,lon)
        Rescaled variable.
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(var) == "DataArray"
    ), 'The input variable "var" is not an xarray.DataArray.'

    # Check if scale is valid.
    scale = scale.lower()
    if scale not in ["daily", "weekly", "monthly", "annual", "seasonal"]:
        raise ValueError("The specified temporal scale is not valid.")

    # Resample variable at selected scale.
    if scale == "daily":
        out = var.resample(time="1D").mean()
    elif scale == "weekly":
        out = var.resample(time="1W").mean()
    elif scale == "monthly":
        out = var.resample(time="1M").mean()
    elif scale == "annual":
        out = var.resample(time="1Y").mean()
    elif scale == "seasonal":
        out = var.resample(time="QS-DEC").mean()
    else:
        out = None

    return out


def select_region(data, latbox, lonbox):
    """
    Select a rectangular region from a DataArray based on a longitude and latitude box.

    Input
    -------
    data: xarray.DataArray ; (time,lat,lon)
        Data array from which the region is selected.
    latbox: tuple / list
        Latitude box (min, max).
    lonbox: tuple / list
        Longitude box (min, max).

    Output
    -------
    out: xarray.DataArray ; (time,lat,lon)
        Data array at selected region.
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(data) == "DataArray"
    ), 'The input variable "data" is not an xarray.DataArray.'
    assert (
        get_type(lonbox) == "tuple" or get_type(lonbox) == "list"
    ), 'The input variable "lonbox" is not a tuple / list.'
    assert (
        get_type(latbox) == "tuple" or get_type(latbox) == "list"
    ), 'The input variable "latbox" is not a tuple / list.'

    # Select the region from the data array.
    out = data.sel(lat=slice(*latbox), lon=slice(*lonbox))

    return out


def select_point(data, target_lon, target_lat):
    """
    Select the closest point from a DataArray based on a longitude and latitude of interest.

    Input
    -------
    data: xarray.DataArray ; (time,lat,lon)
        Data array from which the point is selected.
    target_lon: float
        Longitude of interest.
    target_lat: float
        Latitude of interest.

    Output
    -------
    out: xarray.DataArray ; (time)
        Data array at closest point.
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(data) == "DataArray"
    ), 'The input variable "data" is not an xarray.DataArray.'
    assert (
        get_type(target_lon) == "float"
    ), 'The input variable "target_lon" is not a float.'
    assert (
        get_type(target_lat) == "float"
    ), 'The input variable "target_lat" is not a float.'

    # Select the closest point from the data array.
    out = data.sel(lon=target_lon, lat=target_lat, method="nearest")

    return out


# Other support functions.


def create_dataset(variables, attrs, coords, dims):
    """
    Creates an xarray dataset with the specified variables, coordinates, dimensions and \
    attributes. Inputs are provided as dictionaries.

    Input
    -------
    vars: dict
        A dictionary where keys are the variable names and values are numpy arrays or lists.
    attrs: dict
        A dictionary where keys are the attribute names and values are numpy arrays or lists.
    coords: dict
        A dictionary where keys are the coordinate names and values are numpy arrays or lists.
    dims: tuple / str
        A tuple / str of dimension names in the order they should appear in the dataset.

    Output
    -------
    ds: xarray.Dataset
        An empty xarray Dataset containing the specified variables, coordinates and dimensions.
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(variables) == "dict"
    ), 'The input variable "vars" is not a dictionary.'
    assert get_type(attrs) == "dict", 'The input variable "attrs" is not a dictionary.'
    assert (
        get_type(coords) == "dict"
    ), 'The input variable "coords" is not a dictionary.'
    assert (
        get_type(dims) == "tuple" or get_type(dims) == "str"
    ), 'The input variable "dims" is not a tuple / str.'

    # Create data_vars dictionary
    data_vars = {var_name: (dims, variables[var_name]) for var_name in variables}

    # Create dataset
    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    # Assign attributes
    ds = ds.assign_attrs(attrs)

    return ds


def get_type(x):
    """
    Return the type of the variable in string format. Checks if variable is of the correct type.

    Input
    -------
    x: any type
        Variable to check. 'x' can be any type of variable.

    Output
    -------
    output: str
        Variable type in string format.
    """
    try:
        return type(x).__name__

    except AttributeError:
        return str(type(x)).split("'")[1]


def check_dims(data, n_dims=3):
    """
    Check if the number of dimensions of a xarray data structure is correct.

    Input
    -------
    data: xarray.Dataset / xarray.DataArray / list of xarray.Dataset / list of xarray.DataArray
        Data structure to check.
    n_dims: int
        Number of dimensions. Default is 3.

    Output
    -------
    If the number of dimensions is correct, return None. Otherwise, raise an error.
    """
    if isinstance(data, list):
        for da in data:
            assert get_type(da) in [
                "Dataset",
                "DataArray",
            ], 'The input variable "data" is not an xarray.Dataset or xarray.DataArray.'
            if len(da.dims) != n_dims:
                raise ValueError("The number of dimensions is not correct.")
    else:
        assert get_type(data) in [
            "Dataset",
            "DataArray",
        ], 'The input variable "data" is not an xarray.Dataset or xarray.DataArray.'
        if len(data.dims) != n_dims:
            raise ValueError("The number of dimensions is not correct.")
