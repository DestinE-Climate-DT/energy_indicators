"""
| Destination Earth: Energy Indicators application

Generic, dependency-light array and numerical utilities used throughout
the package, including unit conversion, wind speed computation,
percentiles, and spatial selection. Imported by nearly every other module.
"""

# External libraries
import xarray as xr
import numpy as np
import pandas as pd


## ================ Functions used in operations: ====================
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
    assert get_type(t) == "DataArray", (
        'The input variable "t" is not an xarray.DataArray.'
    )
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
    assert get_type(u) == "DataArray", (
        'The input variable "u" is not an xarray.DataArray.'
    )
    assert get_type(v) == "DataArray", (
        'The input variable "v" is not an xarray.DataArray.'
    )

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
    assert get_type(data) == "DataArray", (
        'The input variable "data" is not an xarray.DataArray.'
    )
    assert get_type(lonbox) == "tuple" or get_type(lonbox) == "list", (
        'The input variable "lonbox" is not a tuple / list.'
    )
    assert get_type(latbox) == "tuple" or get_type(latbox) == "list", (
        'The input variable "latbox" is not a tuple / list.'
    )

    # Get coordinate names (handle both 'lat'/'lon' and 'latitude'/'longitude').
    lat_name = "lat" if "lat" in data.coords else "latitude"
    lon_name = "lon" if "lon" in data.coords else "longitude"

    lat_coords = data[lat_name]
    lon_coords = data[lon_name]

    # Transform longitudes from 0-360 to -180-180 if the box uses negative values.
    if lon_coords.max() > 180 and any(lon < 0 for lon in lonbox):
        data = data.assign_coords({lon_name: (((data[lon_name] + 180) % 360) - 180)})
        data = data.sortby(lon_name)
        lon_coords = data[lon_name]

    # Reverse the latbox order if latitude coordinates are in descending order.
    if lat_coords[0] > lat_coords[-1]:
        lat_slice = slice(latbox[1], latbox[0])
    else:
        lat_slice = slice(latbox[0], latbox[1])

    # Cross-antimeridian box: lonbox[1] > 180 means the right bound is in 0-360
    # notation and wraps past the date line. Split into two -180/180 selections
    # and concatenate.
    if lonbox[1] > 180:
        east = data.sel({lat_name: lat_slice, lon_name: slice(lonbox[0], 180)})
        west = data.sel({lat_name: lat_slice, lon_name: slice(-180, lonbox[1] - 360)})
        return xr.concat([east, west], dim=lon_name).sortby(lon_name)

    # Select the region from the data array.
    out = data.sel({lat_name: lat_slice, lon_name: slice(*lonbox)})

    return out


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


## ======== Functions not used in operations: ==============

def area_weighted_mean(data):
    """
    Compute the area-weighted mean of a variable.

    Input
    -------
    data: xarray.DataArray ; (time,lat,lon)
        Variable.

    Output
    -------
    out: xarray.DataArray ; (time)
        Area-weighted mean of the variable.
    """
    # Check if the input parameters satisfy the required conditions.
    assert get_type(data) == "DataArray", (
        'The input variable "data" is not an xarray.DataArray.'
    )

    # Get coordinate names (handle both 'lat'/'lon' and 'latitude'/'longitude').
    lat_name = "lat" if "lat" in data.coords else "latitude"
    lon_name = "lon" if "lon" in data.coords else "longitude"

    # Compute the latitude-based weights.
    weights = np.cos(np.deg2rad(data[lat_name]))

    # Compute the area-weighted mean.
    out = data.weighted(weights).mean(dim=(lat_name, lon_name))

    return out


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


def check_radiation(data):
    """
    Check if radiation is in J/m2 or W/m2.

    Input
    -------
    data: xarray.Dataset / xarray.DataArray
        Radiation data.

    Output
    -------
    unit: str
        Radiation unit of the data.
        Possible values are 'J/m2' and 'W/m2'.
    """
    # Check if the input parameters satisfy the required conditions.
    assert get_type(data) in [
        "Dataset",
        "DataArray",
    ], 'The input variable "data" is not an xarray.Dataset / xarray.DataArray.'

    # Check if radiation is in J/m2 or W/m2.
    try:
        raw_unit = data.attrs["units"].strip()
        if raw_unit in ["J/m2", "J m**-2"]:
            unit = "J/m2"
        elif raw_unit in ["W/m2", "W m**-2"]:
            unit = "W/m2"
        else:
            raise ValueError("The radiation unit is not valid.")
    except AttributeError:
        raise AttributeError('The data object does not have the attribute "units".')

    return unit


def convert_radiation(rsds):
    """
    Convert radiation from J/m2 to W/m2 by dividing by 3600 seconds.

    Input
    -------
    rsds: xarray.DataArray ; (time,lat,lon)
        Radiation (in J/m2 or W/m2).

    Output
    -------
    r_conv: xarray.DataArray ; (time,lat,lon)
        Radiation (in W/m2).
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(rsds) == "DataArray"
    ), 'The input variable "rsds" is not an xarray.DataArray.'

    # Convert only if the unit is J/m2 or J m**-2.
    try:
        raw_unit = rsds.attrs["units"].strip()
        if raw_unit in ["J/m2", "J m**-2"]:
            r_conv = rsds / 3600
            r_conv.attrs["units"] = "W/m2"
        else:
            r_conv = rsds
    except AttributeError:
        raise AttributeError('The data object does not have the attribute "units".')

    return r_conv


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
    -----------
    [1]: https://doi.org/10.1002/2015GL066868
    """
    # Check if the input parameters satisfy the required conditions.
    assert get_type(start_date) == "datetime64", (
        'The input variable "start_date" is not a numpy.datetime64.'
    )
    assert get_type(end_date) == "datetime64", (
        'The input variable "end_date" is not a numpy.datetime64.'
    )
    assert get_type(lats) == "ndarray", (
        'The input variable "lats" is not a numpy.ndarray.'
    )
    assert get_type(lons) == "ndarray", (
        'The input variable "lons" is not a numpy.ndarray.'
    )

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
    assert get_type(data) == "DataArray", (
        'The input variable "data" is not an xarray.DataArray.'
    )
    assert get_type(target_lon) in ["int", "float"], (
        'The input variable "target_lon" is not a float / int.'
    )
    assert get_type(target_lat) in ["int", "float"], (
        'The input variable "target_lat" is not a float / int.'
    )

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
    variables: dict
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
        An xarray Dataset containing the specified variables, coordinates and dimensions.
    """
    # Check if the input parameters satisfy the required conditions.
    assert get_type(variables) == "dict", (
        'The input variable "vars" is not a dictionary.'
    )
    assert get_type(attrs) == "dict", 'The input variable "attrs" is not a dictionary.'
    assert get_type(coords) == "dict", (
        'The input variable "coords" is not a dictionary.'
    )
    assert get_type(dims) == "tuple" or get_type(dims) == "str", (
        'The input variable "dims" is not a tuple / str.'
    )

    # Create the dataset with data variables, coordinates and attributes in one step.
    ds = xr.Dataset(
        data_vars={var_name: (dims, variables[var_name]) for var_name in variables},
        coords=coords,
        attrs=attrs,
    )

    return ds
