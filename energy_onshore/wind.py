#!/usr/bin/env python3
"""
# Destination Earth: Energy Indicators application
# Authors: Aleksander Lacma-Nadolnik, Francesc Roura-Adserias, Sushovan Ghosh
# Version: 1.1.x
"""

# External libraries
import os
import xarray as xr
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator
from one_pass.opa import Opa
import glob

# Internal libraries   
from energy_onshore.utils import load_turbines

from .core import get_type, select_region, wind_speed, select_point
from .mask_processing import onshore_mask

# Development of wind energy indicators for the Energy Indicators application.

def wind_direction(u, v, mask=None):
    """
    Compute direction from where wind is blowing from u and v components.

    Input
    -------
    u: xarray.DataArray ; (time,lat,lon)
        U-component of wind.
    v: xarray.DataArray ; (time,lat,lon)
        V-component of wind.
    mask: str or None (None by default).
        Path to a netCDF file containing a land-sea mask. If provided, the mask will be applied.

    Output
    -------
    wd: xarray.DataArray ; (time,lat,lon)
        Wind direction in degrees.
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(u) == "DataArray"
    ), 'The input variable "u" is not an xarray.DataArray.'
    assert (
        get_type(v) == "DataArray"
    ), 'The input variable "v" is not an xarray.DataArray.'
    # Check the dimensions of the input variables.
    for var in [u, v]:
        assert (
            var.ndim == 3
        ), f'The input variable "{var.name}" does not have the required dimensions (time, lat,\
        lon).'
    
    # Apply the land-sea mask by default.
    if mask is not None:
        u = onshore_mask(u, mask)
        v = onshore_mask(v, mask)
    if "level" in u.dims:
        u = u.squeeze("level", drop=True)
    if "level" in v.dims:
        v = v.squeeze("level", drop=True)

    # Compute wind direction.
    wind_dir = np.arctan2(-u, -v)

    # Convert wind direction from radians to degrees.
    wind_dir = np.degrees(wind_dir)

    # Ensure wind direction is in the range [0, 360).
    wind_dir = (wind_dir + 360) % 360

    # Add metadata to the output variable.
    attrs = {"shortname": "wind_dir", "longname": "Wind direction", "units": "degrees"}
    coords = {"time": u.time, "lat": u.lat, "lon": u.lon}
    dims = ("time", "lat", "lon")

    wind_dir = xr.DataArray(
        wind_dir, dims=dims, coords=coords, attrs=attrs, name=attrs["shortname"]
    )

    return wind_dir


def wind_speed_anomalies(ws, climatology, scale="daily"):
    """
    Compute the wind speed anomalies.

    Input
    -------
    ws: xarray.DataArray ; (time,lat,lon)
        Wind speed magnitude.
    climatology: xarray.DataArray ; (time,lat,lon)
        Climatology of the wind speed for a 30-year reference period. Frequency of the \
        climatology must be the same as the frequency of the input variable "ws" and at least \
        'daily'.
    scale: str
        Temporal scale of the climatology. Options are 'daily', 'monthly' and 'yearly'. \
        (default: 'daily')

    Output
    -------
    ws_anom: xarray.DataArray ; (time,lat,lon)
        Wind speed anomalies.

    References
    -------
    [1]: https://doi.org/10.1175/JCLI3366.1
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(ws) == "DataArray"
    ), 'The input variable "ws" is not an xarray.DataArray.'
    assert (
        get_type(climatology) == "DataArray"
    ), 'The input variable "climatology" is not an xarray.DataArray.'
    # Check the dimensions of the input variables.
    for var in [ws, climatology]:
        assert (
            var.ndim == 3
        ), f'The input variable "{var.name}" does not have the required dimensions (time, lat,\
        lon).'

    # Extract boundaries of the region of interest.
    latmin = np.round(ws.lat[0].values, 2)
    latmax = np.round(ws.lat[-1].values, 2)
    lonmin = np.round(ws.lon[0].values, 2)
    lonmax = np.round(ws.lon[-1].values, 2)

    # Select corresponding region from climatology.
    latbox = [latmin, latmax]
    lonbox = [lonmin, lonmax]
    clim_reg = select_region(climatology, latbox, lonbox)

    # Extract time period from ws.
    if ws.indexes["time"].is_leap_year.any():
        tvec = ws["time.dayofyear"]
        # Filter out February 29th.
        ws = ws.sel(time=~((ws.time.dt.month == 2) & (ws.time.dt.day == 29)))
        leap_day_index = 60
        if tvec.max() > leap_day_index and tvec.min() <= leap_day_index:
            tvec_inf = tvec.where(tvec <= leap_day_index, drop=True)
            tidx_inf = tvec_inf.values - 1
            tvec_sup = tvec.where(tvec > leap_day_index, drop=True)
            tidx_sup = np.where(tvec_sup > leap_day_index, tvec_sup - 1, tvec_sup)
            tidx = np.concatenate([tidx_inf, tidx_sup[:-1]])
            tidx = tidx.astype(int)
        elif tvec.max() <= leap_day_index and tvec.min() <= leap_day_index:
            tvec = tvec.where(tvec != leap_day_index, drop=True)
            tidx = tvec.values - 1
            tidx = tidx.astype(int)
        elif tvec.max() > leap_day_index and tvec.min() > leap_day_index:
            tidx = tvec.values - 2
            tidx = tidx.astype(int)
    else:
        tvec = ws["time.dayofyear"]
        tidx = tvec.values - 1
        tidx = tidx.astype(int)

    # Select corresponding time period from climatology.
    clim = clim_reg[tidx, :, :]

    # Define resampling frequencies.
    freq_dict = {"daily": "1D", "monthly": "1M", "yearly": "1Y"}
    freq = freq_dict.get(scale, -1)

    # Resample and compute wind speed anomalies.
    ws_resampled = ws.resample(time=freq).mean()
    clim_resampled = clim.resample(time=freq).mean()
    ws_anom = ws_resampled - np.array(clim_resampled)

    # Add metadata to the output variable.
    attrs = {"shortname": "ws_anom", "longname": "Wind speed anomalies", "units": "m/s"}
    coords = {"time": ws_anom.time, "lat": ws_anom.lat, "lon": ws_anom.lon}
    dims = ("time", "lat", "lon")

    ws_anom = xr.DataArray(
        ws_anom, coords=coords, dims=dims, attrs=attrs, name=attrs["shortname"]
    )

    return ws_anom


def wind_power_density(ws, air_density=1.225, mask=None):
    """
    Compute the wind power density.

    Input
    -------
    ws: xarray.DataArray ; (time,lat,lon)
        Wind speed magnitude.
    air_density: float or xarray.DataArray ; (time,lat,lon)
        Air density. Default value is 1.225 kg m^(-3).
    mask: str or None (None by default).
        Path to a netCDF file containing a land-sea mask. If provided, the mask will be applied.

    Output
    -------
    wpd: xarray.DataArray ; (time,lat,lon)
        Wind power density.

    References
    -------
    [1]: https://doi.org/10.1016/j.rser.2009.07.028
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(ws) == "DataArray"
    ), 'The input variable "ws" is not an xarray.DataArray.'
    if get_type(air_density) != "float":
        assert (
            get_type(air_density) == "DataArray"
        ), 'The input variable "air_density" is not an xarray.DataArray.'
        assert (
            air_density.ndim == 3
        ), 'The input variable "air_density" does not have the required dimensions (time,lat,lon).'
    # Check the dimensions of the input variables.
    assert (
        ws.ndim == 3
    ), 'The input variable "ws" does not have the required dimensions (time,lat,lon).'

    # Apply the land-sea mask by default.
    if mask is not None:
        ws = onshore_mask(ws, mask)
    if "level" in ws.dims:
        ws = ws.squeeze("level", drop=True)

    # Compute the wind power density.
    wpd = 0.5 * air_density * ws**3

    # Add metadata to the output variable.
    attrs = {"shortname": "wpd", "longname": "Wind power density", "units": "W/m^2"}
    coords = {"time": ws.time, "lat": ws.lat, "lon": ws.lon}
    dims = ("time", "lat", "lon")

    wpd = xr.DataArray(
        wpd, coords=coords, dims=dims, attrs=attrs, name=attrs["shortname"]
    )

    return wpd


def wind_speed_at_height(u10, v10, hub_height, alpha=0.143):
    """
    Compute the wind speed at a given hub height using an exponential wind profile.
    Only required if wind components are provided at 10 m.

    Input
    -------
    u10: xarray.DataArray ; (time,lat,lon)
        Zonal wind component at 10m.
    v10: xarray.DataArray ; (time,lat,lon)
        Meridional wind component at 10m.
    hub_height: float
        Height at which the wind speed is computed.
    alpha: float
        Wind shear exponent (default: 0.143 [1]). For offshore wind farms, alpha=0.11 [2].

    Output
    -------
    wsh: xarray.DataArray ; (time,lat,lon)
        Wind speed at the given height.

    References
    -------
    [1]: https://doi.org/10.1080/00022470.1977.10470503
    [2]: https://doi.org/10.1175/1520-0450(1994)033<0757:DTPLWP>2.0.CO;2
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(u10) == "DataArray"
    ), 'The input variable "u10" is not an xarray.DataArray.'
    assert (
        get_type(v10) == "DataArray"
    ), 'The input variable "v10" is not an xarray.DataArray.'
    assert (
        get_type(hub_height) == "float"
    ), 'The input variable "height" is not a float.'
    assert get_type(alpha) == "float", 'The input variable "alpha" is not a float.'
    # Check the dimensions of the input variables.
    for var in [u10, v10]:
        assert (
            var.ndim == 3
        ), f"The input variable {var.name} does not have the required dimensions (time,lat,lon)."

    # Compute the wind speed at the given height.
    ws = wind_speed(u10, v10)
    wshh = ws * (hub_height / 10) ** alpha

    # Add metadata to the output variable.
    attrs = {
        "shortname": "wshh",
        "longname": "Wind speed at hub height",
        "units": "m/s",
    }
    coords = {"time": ws.time, "lat": ws.lat, "lon": ws.lon}
    dims = ("time", "lat", "lon")

    wshh = xr.DataArray(
        wshh, coords=coords, dims=dims, attrs=attrs, name=attrs["shortname"]
    )

    return wshh


def power_output(ws, iec_class, rated_power, cut_in_speed, rated_speed, cut_out_speed):
    """
    Compute the estimated power output of a wind turbine (PCHIP interpolator).

    Input
    -------
    ws: numpy.ndarray ; (time * lat * lon)
        Wind speed magnitude at hub height (flattened array).
    iec_class: str
        IEC wind turbine class. Options are 'I','I/II', 'II', 'II/III', 'III', 'S'.
    rated_power: int
        Rated power of the wind turbine in kW.
    cut_in_speed: float
        Cut-in speed of the wind turbine in m s^(-1).
    rated_speed: float
        Rated speed of the wind turbine in m s^(-1).
    cut_out_speed: float
        Cut-out speed of the wind turbine in m s^(-1).

    Output
    -------
    power_out: xarray.DataArray ; (time * lat * lon)
        Power output in kW (flattened array).

    References
    -------
    [1]: https://doi.org/10.1016/j.rser.2009.07.028
    [2]: https://doi.org/10.3389/fenrg.2021.769920
    [3]: https://www.thewindpower.net/turbine_es_5_enercon_e70-2300.php
    [4]: https://www.thewindpower.net/turbine_es_44_gamesa_g80-2000.php
    [5]: https://www.thewindpower.net/turbine_es_46_gamesa_g87-2000.php
    [6]: https://www.thewindpower.net/turbine_es_779_vestas_v100-2000.php
    [7]: https://www.thewindpower.net/turbine_es_590_vestas_v110-2000.php
    [8]: https://www.thewindpower.net/turbine_es_1476_vestas_v164-9500.php
    [9]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    """
    # Check if the input parameters satisfy the required conditions.
    if not isinstance(rated_power, (float, int)):
        raise TypeError(
            'The input variable "rated_power" must be a float or an integer.'
        )
    assert get_type(ws) == "ndarray", 'The input variable "ws" is not a numpy.ndarray.'
    assert (
        get_type(cut_in_speed) == "float"
    ), 'The input variable "cut_in_speed" is not a float.'
    assert (
        get_type(rated_speed) == "float"
    ), 'The input variable "rated_speed" is not a float.'
    assert (
        get_type(cut_out_speed) == "float"
    ), 'The input variable "cut_out_speed" is not a float.'
    # Check if the IEC class is valid.
    if iec_class not in ["I", "I_II", "II", "II_III", "III", "S"]:
        raise ValueError(
            'The IEC class is not valid. Options are "I","I/II", "II", "II/III", "III", "S".'
        )

    iec_to_t = {"I": 0, "I_II": 1, "II": 2, "II_III": 3, "III": 4, "S": 5}
    t = iec_to_t.get(iec_class, -1)

    turbines = [
        "Enercon_E70_2.3MW",
        "Gamesa_G80_2.0MW",
        "Gamesa_G87_2.0MW",
        "Vestas_V100_2.0MW",
        "Vestas_V110_2.0MW",
        "Vestas_V164_9.5MW",
    ]

    # Read the csv file with the manufacturer power curve.
    turbine_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "power_curves", f"{turbines[t]}.csv"
    )
    power_curve_data = pd.read_csv(turbine_path, skiprows=1, sep=",")

    # Extract wind speed and power data
    x_data = power_curve_data.wind_speed.values
    y_data = power_curve_data.power.values

   # Sort data by wind speed to ensure proper interpolation
    sort_idx = np.argsort(x_data)
    x_data_sorted = x_data[sort_idx]
    y_data_sorted = y_data[sort_idx]

    # Create PCHIP interpolator from manufacturer data
    pchip_interpolator = PchipInterpolator(x_data_sorted, y_data_sorted)

    # Define conditions for piecewise function
    conditions = [
        (ws < cut_in_speed),
        (ws >= cut_in_speed) & (ws <= rated_speed),
        (ws > rated_speed) & (ws <= cut_out_speed),
        (ws > cut_out_speed),
    ]

    # Use PCHIP interpolator to compute power output over operational region
    def interpolated_power(x):
        interpolated = pchip_interpolator(x)

        # Ensure power does not exceed rated power
        power_interp = np.minimum(interpolated, rated_power)

        return power_interp

    # Define corresponding functions for piecewise function
    functions = [
        0,
        interpolated_power,
        rated_power,
        0,
    ]

    # A piecewise function is used to obtain the power output for the entire wind speed range.
    power_out = np.piecewise(ws, conditions, functions)

    # Store the power curve and the wind values in an xarray.DataArray.
    attrs = {"shortname": "po", "longname": "Power output", "units": "kW"}
    coords = {"wind_speed": ws}
    dims = "wind_speed"

    power_out = xr.DataArray(
        power_out, coords=coords, dims=dims, attrs=attrs, name=attrs["shortname"]
    )

    return power_out


def capacity_factor(ws, iec_class, mask=None):
    """
    Compute the capacity factor of a wind turbine.

    Input
    -------
    ws: xarray.DataArray ; (time,lat,lon)
        Wind speed magnitude at hub height.
    iec_class: str
        IEC wind turbine class. Options are 'I','I/II', 'II', 'II/III', 'III', 'S'.
    mask: str or None (None by default)
        Path to a netCDF file containing a land-sea mask. If provided, the mask will be applied.

    Output
    -------
    cf: xarray.DataArray ; (time,lat,lon)
        Capacity factor.

    References
    -------
    [1]: https://doi.org/10.1016/j.renene.2019.04.135
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(ws) == "DataArray"
    ), 'The input variable "ws" is not an xarray.DataArray.'
    # Check the dimensions of the input variables.
    assert (
        ws.ndim == 3
    ), 'The input variable "ws" does not have the required dimensions (time,lat,lon).'

    # Check if the IEC class is valid.
    if iec_class not in ["I", "I_II", "II", "II_III", "III", "S"]:
        raise ValueError(
            "Input argument 'iec_class' must be 'I','I/II','II','II/III', 'III' or 'S'."
        )

    #load available turbines' specifications
    turbine_class = load_turbines()

    iec_to_t = {"I": 0, "I_II": 1, "II": 2, "II_III": 3, "III": 4, "S": 5}
    t = iec_to_t.get(iec_class, -1)

    # Compute the power curve of the wind turbine.
    rated_power = turbine_class[t]["rated_power"]
    cut_in_speed = turbine_class[t]["cut_in_speed"]
    rated_speed = turbine_class[t]["rated_speed"]
    cut_out_speed = turbine_class[t]["cut_out_speed"]

    # Apply the land-sea mask by default.
    if mask is not None:
        ws = onshore_mask(ws, mask)
    if "level" in ws.dims:
        ws = ws.squeeze("level", drop=True)

    # Flatten the wind array
    ws_flatten = np.array(ws).flatten()

    # Obtain the power output
    power_out_flatten = power_output(
        ws_flatten, iec_class, rated_power, cut_in_speed, rated_speed, cut_out_speed
    )

    # Reshape power output to match original wind speed input array.
    power_out = np.reshape(np.array(power_out_flatten), ws.shape)

    # Compute the capacity factor.
    cf = power_out / rated_power

    # Add metadata to the output variable.
    attrs = {"shortname": f'cf_{iec_class.lower()}', \
        "longname": f"Capacity factor for turbine type {iec_class}", "units": "-"}
    coords = {"time": ws.time, "lat": ws.lat, "lon": ws.lon}
    dims = ("time", "lat", "lon")

    cf = xr.DataArray(
        cf, coords=coords, dims=dims, attrs=attrs, name=attrs["shortname"]
    )

    return cf.to_dataset()

# Warning: deprecated. CF histograms are computed via the OPA implementation
# under capacity_factor_histogram_opa().
def capacity_factor_histogram(ws, bins, iec_class):
    """
    Compute the capacity factor histogram of a wind turbine over a 2D grid.

    Input
    -------
    ws: xarray.DataArray ; (time,lat,lon)
        Wind speed magnitude at hub height.
    bins: int
        Number of bins.
    iec_class: str
        IEC wind turbine class. Options are 'I','I/II', 'II', 'II/III', 'III', 'S'.

    Output
    -------
    counts: xarray.DataArray ; (bins,lat,lon)
        Histogram counts.
    bin_edges: xarray.DataArray ; (bins+1,lat,lon)
        Bin edges.

    References
    -------
    [1]: https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html
    [2]: https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(ws) == "DataArray"
    ), 'The input variable "ws" is not an xarray.DataArray.'
    assert get_type(bins) == "int", 'The input variable "bins" is not an integer.'
    # Check if the IEC class is valid.
    if iec_class not in ["I", "I_II", "II", "II_III", "III", "S"]:
        raise ValueError(
            "Input argument 'iec_class' must be 'I','I/II','II','II/III', 'III' or 'S'."
        )
    # Check the dimensions of the input variables.
    assert (
        ws.ndim == 3
    ), 'The input variable "ws" does not have the required dimensions (time,lat,lon).'

    #load available turbines' specifications
    turbine_class = load_turbines()

    iec_to_t = {"I": 0, "I_II": 1, "II": 2, "II_III": 3, "III": 4, "S": 5}
    t = iec_to_t.get(iec_class, -1)

    # Compute the power curve of the wind turbine.
    rated_power = turbine_class[t]["rated_power"]
    cut_in_speed = turbine_class[t]["cut_in_speed"]
    rated_speed = turbine_class[t]["rated_speed"]
    cut_out_speed = turbine_class[t]["cut_out_speed"]

    # Flatten the wind array
    ws_flatten = np.array(ws).flatten()

    # Obtain the power output
    power_out_flatten = power_output(
        ws_flatten, iec_class, rated_power, cut_in_speed, rated_speed, cut_out_speed
    )

    # Reshape power output to match original wind speed input array.
    power_out = np.reshape(np.array(power_out_flatten), ws.shape)

    # Compute the capacity factor.
    cf = power_out / rated_power

    # Compute the capacity factor histogram.
    def compute_histogram_counts(x):
        counts = np.histogram(x, bins=bins)[0]
        return counts

    def compute_histogram_bin_edges(x):
        bin_edges = np.histogram(x, bins=bins)[1]
        return bin_edges

    counts, bin_edges = np.apply_along_axis(compute_histogram, axis=0, arr=cf)

    # Add metadata to the output variable.
    attrs = {"shortname": "counts", "longname": "Histogram counts", "units": "-"}
    attrs_bin = {"shortname": "bin_edges", "longname": "Bin edges", "units": "-"}
    coords = {"lat": ws.lat, "lon": ws.lon}
    dims = ("lat", "lon")

    counts = xr.DataArray(
        counts,
        coords=coords,
        dims=dims,
        attrs=attrs,
        name=attrs["shortname"],
    )
    bin_edges = xr.DataArray(
        bin_edges,
        coords=coords,
        dims=dims,
        attrs=attrs_bin,
        name=attrs_bin["shortname"],
    )

    return counts, bin_edges


def capacity_factor_histogram_1d(ws, bins, target_lon, target_lat, iec_class):
    """
    Compute the capacity factor histogram of a wind turbine for a given location.

    Input
    -------
    ws: xarray.DataArray ; (time,lat,lon)
        Wind speed magnitude at hub height.
    bins: int
        Number of bins.
    target_lon: float
        Longitude of the target location.
    target_lat: float
        Latitude of the target location.
    iec_class: str
        IEC wind turbine class. Options are 'I','I/II', 'II', 'II/III', 'III', 'S'.

    Output
    -------
    counts: xarray.DataArray ; (bins)
        Histogram counts.
    bin_edges: xarray.DataArray ; (bins+1)
        Bin edges.

    References
    -------
    [1]: https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(ws) == "DataArray"
    ), 'The input variable "ws" is not an xarray.DataArray.'
    assert get_type(bins) == "int", 'The input variable "bins" is not an integer.'
    assert (
        get_type(target_lon) == "float"
    ), 'The input variable "target_lon" is not a float.'
    assert (
        get_type(target_lat) == "float"
    ), 'The input variable "target_lat" is not a float.'
    # Check if the IEC class is valid.
    if iec_class not in ["I", "I_II", "II", "II_III", "III", "S"]:
        raise ValueError(
            "Input argument 'iec_class' must be 'I','I/II','II','II/III', 'III' or 'S'."
        )
    # Check the dimensions of the input variables.
    assert (
        ws.ndim == 3
    ), 'The input variable "ws" does not have the required dimensions (time,lat,lon).'

    #load available turbines' specifications
    turbine_class = load_turbines()

    iec_to_t = {"I": 0, "I_II": 1, "II": 2, "II_III": 3, "III": 4, "S": 5}
    t = iec_to_t.get(iec_class, -1)

    # Compute the power curve of the wind turbine.
    rated_power = turbine_class[t]["rated_power"]
    cut_in_speed = turbine_class[t]["cut_in_speed"]
    rated_speed = turbine_class[t]["rated_speed"]
    cut_out_speed = turbine_class[t]["cut_out_speed"]

    # Flatten the wind array
    ws_flatten = np.array(ws).flatten()

    # Obtain the power output
    power_out_flatten = power_output(
        ws_flatten, iec_class, rated_power, cut_in_speed, rated_speed, cut_out_speed
    )

    # Reshape power output to match original wind speed input array.
    power_out = np.reshape(np.array(power_out_flatten), ws.shape)

    # Compute the capacity factor.
    cf = power_out / rated_power

    # Compute the capacity factor histogram.
    counts, bin_edges = np.histogram(cf, bins=bins)

    # Add metadata to the output variable.
    attrs = {"shortname": "counts", "longname": "Histogram counts", "units": "-"}
    attrs_bin = {"shortname": "bin_edges", "longname": "Bin edges", "units": "-"}
    coords = {"bins": np.arange(bins)}
    coords_bin = {"bins": np.arange(bins + 1)}
    dims = "bins"

    counts = xr.DataArray(
        counts,
        coords=coords,
        dims=dims,
        attrs=attrs,
        name=attrs["shortname"],
    )
    bin_edges = xr.DataArray(
        bin_edges,
        coords=coords_bin,
        dims=dims,
        attrs=attrs_bin,
        name=attrs_bin["shortname"],
    )

    return counts, bin_edges


def wind_speed_histogram(ws, bins):
    """
    Compute the wind speed histogram over a 2D grid.

    Input
    -------
    ws: xarray.DataArray ; (time,lat,lon)
        Wind speed magnitude at hub height.
    bins: int
        Number of bins.

    Output
    -------
    counts: xarray.DataArray ; (bins,lat,lon)
        Histogram counts.
    bin_edges: xarray.DataArray ; (bins+1,lat,lon)
        Bin edges.

    References
    -------
    [1]: https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html
    [2]: https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(ws) == "DataArray"
    ), 'The input variable "ws" is not an xarray.DataArray.'
    assert get_type(bins) == "int", 'The input variable "bins" is not an integer.'
    # Check the dimensions of the input variables.
    assert (
        ws.ndim == 3
    ), 'The input variable "ws" does not have the required dimensions (time,lat,lon).'

    # Compute the wind speed histogram.
    def compute_histogram_counts(x):
        counts = np.histogram(x, bins=bins)[0]
        return counts

    def compute_histogram_bin_edges(x):
        bin_edges = np.histogram(x, bins=bins)[1]
        return bin_edges

    counts = np.apply_along_axis(compute_histogram_counts, axis=0, arr=ws)
    bin_edges = np.apply_along_axis(compute_histogram_bin_edges, axis=0, arr=ws)

    # Add metadata to the output variable.
    attrs = {"shortname": "counts", "longname": "Histogram counts", "units": "-"}
    attrs_bin = {"shortname": "bin_edges", "longname": "Bin edges", "units": "-"}
    coords = {"lat": ws.lat, "lon": ws.lon}
    dims = ("lat", "lon")

    counts = xr.DataArray(
        counts, coords=coords,
        dims=dims,
        attrs=attrs,
        name=attrs["shortname"],
    )
    bin_edges = xr.DataArray(
        bin_edges,
        coords=coords,
        dims=dims,
        attrs=attrs_bin,
        name=attrs_bin["shortname"],
    )

    return counts, bin_edges


def wind_speed_histogram_1d(ws, bins, target_lon, target_lat):
    """
    Compute the wind speed histogram for a given location.

    Input
    -------
    ws: xarray.DataArray ; (time,lat,lon)
        Wind speed magnitude at hub height.
    bins: int
        Number of bins.
    target_lon: float
        Longitude of the target location.
    target_lat: float
        Latitude of the target location.

    Output
    -------
    counts: xarray.DataArray ; (bins)
        Histogram counts.
    bin_edges: xarray.DataArray ; (bins+1)
        Bin edges.

    References
    -------
    [1]: https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(ws) == "DataArray"
    ), 'The input variable "ws" is not an xarray.DataArray.'
    assert get_type(bins) == "int", 'The input variable "bins" is not an integer.'
    assert (
        get_type(target_lon) == "float"
    ), 'The input variable "target_lon" is not a float.'
    assert (
        get_type(target_lat) == "float"
    ), 'The input variable "target_lat" is not a float.'
    # Check the dimensions of the input variables.
    assert (
        ws.ndim == 3
    ), 'The input variable "ws" does not have the required dimensions (time,lat,lon).'

    # Select the wind speed values at the target location.
    ws_target = select_point(ws, target_lon, target_lat)

    # Compute the wind speed histogram.
    counts, bin_edges = np.histogram(ws_target, bins=bins)

    # Add metadata to the output variable.
    attrs = {"shortname": "counts", "longname": "Histogram counts", "units": "-"}
    attrs_bin = {"shortname": "bin_edges", "longname": "Bin edges", "units": "-"}
    coords = {"bins": np.arange(bins)}
    coords_bin = {"bin_edges": np.arange(bins + 1)}
    dims = "bins"
    dims_bin = "bin_edges"

    counts = xr.DataArray(
        counts,
        coords=coords,
        dims=dims,
        attrs=attrs,
        name=attrs["shortname"],
    )
    bin_edges = xr.DataArray(
        bin_edges,
        coords=coords_bin,
        dims=dims_bin,
        attrs=attrs_bin,
        name=attrs_bin["shortname"],
    )

    return counts, bin_edges

def annual_energy_production_wind(capacity_factor, rated_power, num_turbines=1):
    """
    Compute the annual energy production of a wind turbine from its capacity factor time series.

    Input
    -------
    capacity_factor_series: xarray.DataArray ; (time)
        Capacity factor time series for a year at 1-hourly frequency.
    rated_power: float / int
        Rated power of the wind turbine in MW.
    num_turbines: int
        Number of wind turbines (default: 1).

    Output
    -------
    aep: xarray.DataArray ; (time)
        Annual energy production in MWh.

    References
    -------
    [1]: https://doi.org/10.1016/j.renene.2015.10.006
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(capacity_factor) == "DataArray"
    ), 'The input variable "capacity_factor_series" is not an xarray.DataArray.'
    if not isinstance(rated_power, (float, int)):
        raise TypeError(
            'The input variable "rated_power" must be a float or an integer.'
        )
    if not isinstance(num_turbines, int):
        raise TypeError('The input variable "num_turbines" must be an integer.')
    # Check the dimensions of the input variables.
    assert (
        capacity_factor.ndim == 1
    ), 'The input variable "capacity_factor" does not have the required dimensions (time).'

    # Compute the annual energy production.
    aep = capacity_factor * rated_power * num_turbines

    # Add metadata to the output variable.
    attrs = {"shortname": "aep", "longname": "Annual energy production", "units": "MWh"}
    coords = {"time": capacity_factor.time}
    dims = "time"

    aep = xr.DataArray(
        aep, coords=coords, dims=dims, attrs=attrs, name=attrs["shortname"]
    )

    return aep


def high_wind_events(ws, threshold=25.0, mask=None):
    """
    Compute where and when wind speed exceeds a given threshold (cut-out speed).

    Input
    -------
    ws: xarray.DataArray ; (time,lat,lon)
        Wind speed magnitude at hub height.
    threshold: float / int
        Wind speed threshold (default: 25.0 m/s). Cut-out speed of the wind turbine.
    mask: str or None (None by default)
        Path to a netCDF file containing a land-sea mask. If provided, the mask will be applied.

    Output
    -------
    hwe: xarray.DataArray ; (lat,lon)
        Number of high wind events.

    References
    -------
    [1]: https://iopscience.iop.org/article/10.1088/1748-9326/acbdb2
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(ws) == "DataArray"
    ), 'The input variable "ws" is not an xarray.DataArray.'
    if not isinstance(threshold, (float, int)):
        raise TypeError('The input variable "threshold" must be a float or an integer.')
    # Check the dimensions of the input variables.
    assert (
        ws.ndim == 3
    ), 'The input variable "ws" does not have the required dimensions (time,lat,lon).'

    # Apply the land-sea mask by default.
    if mask is not None:
        ws = onshore_mask(ws, mask)
    if "level" in ws.dims:
        ws = ws.squeeze("level", drop=True)

    # Compute the number of high wind events.
    hwe = np.sum(ws > threshold, axis=0)

    # Add metadata to the output variable.
    attrs = {"shortname": "hwe", "longname": "Number of high wind events", "units": "-"}
    coords = {"lat": ws.lat, "lon": ws.lon}
    dims = ("lat", "lon")

    hwe = xr.DataArray(
        hwe, coords=coords, dims=dims, attrs=attrs, name=attrs["shortname"]
    )

    return hwe.to_dataset()


def low_wind_events(ws, threshold=3.0, mask=None):
    """
    Compute where and when wind speed is below a given threshold (cut-in speed).

    Input
    -------
    ws: xarray.DataArray ; (time,lat,lon)
        Wind speed magnitude at hub height.
    threshold: float / int
        Wind speed threshold (default: 3.0 m/s). Cut-in speed of the wind turbine.
    mask: str or None (None by default)
        Path to a netCDF file containing a land-sea mask. If provided, the mask will be applied.

    Output
    -------
    lwe: xarray.DataArray ; (lat,lon)
        Number of low wind events.

    References
    -------
    [1]: https://iopscience.iop.org/article/10.1088/1748-9326/acbdb2
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(ws) == "DataArray"
    ), 'The input variable "ws" is not an xarray.DataArray.'
    if not isinstance(threshold, (float, int)):
        raise TypeError('The input variable "threshold" must be a float or an integer.')
    # Check the dimensions of the input variables.
    assert (
        ws.ndim == 3
    ), 'The input variable "ws" does not have the required dimensions (time,lat,lon).'

    # Apply the land-sea mask by default.
    if mask is not None:
        ws = onshore_mask(ws, mask)
    if "level" in ws.dims:
        ws = ws.squeeze("level", drop=True)

    # Compute the number of low wind events.
    lwe = np.sum(ws < threshold, axis=0)

    # Add metadata to the output variable.
    attrs = {"shortname": "lwe", "longname": "Number of low wind events", "units": "-"}
    coords = {"lat": ws.lat, "lon": ws.lon}
    dims = ("lat", "lon")

    lwe = xr.DataArray(
        lwe, coords=coords, dims=dims, attrs=attrs, name=attrs["shortname"]
    )

    return lwe.to_dataset()


def calm_days(ws, threshold=2.0, mask=None):
    """
    Compute where and when daily average wind speed is below a given threshold (calm days).

    Input
    -------
    ws: xarray.DataArray ; (time,lat,lon)
        Daily wind speed magnitude.
    threshold: float / int
        Wind speed threshold (default: 2.0 m/s). Calm wind speed threshold.
    mask: str or None (None by default)
        Path to a netCDF file containing a land-sea mask. If provided, the mask will be applied.

    Output
    -------
    cd: xarray.DataArray ; (lat,lon)
        Number of calm days.

    References
    -------
    [1]: https://xclim.readthedocs.io/en/stable/indices.html#xclim.indices.calm_days
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(ws) == "DataArray"
    ), 'The input variable "ws" is not an xarray.DataArray.'
    if not isinstance(threshold, (float, int)):
        raise TypeError('The input variable "threshold" must be a float or an integer.')
    # Check the dimensions of the input variables.
    assert (
        ws.ndim == 3
    ), 'The input variable "ws" does not have the required dimensions (time,lat,lon).'

    # Apply the land-sea mask by default.
    if mask is not None:
        ws = onshore_mask(ws, mask)
    if "level" in ws.dims:
        ws = ws.squeeze("level", drop=True)

    # Compute the number of calm days.

    cd = np.sum(ws < threshold, axis=0)

    # Add metadata to the output variable.
    attrs = {"shortname": "cd", "longname": "Number of calm days", "units": "-"}
    coords = {"lat": ws.lat, "lon": ws.lon}
    dims = ("lat", "lon")

    cd = xr.DataArray(
        cd, coords=coords, dims=dims, attrs=attrs, name=attrs["shortname"]
    )

    return cd.to_dataset()


def windy_days(ws, threshold=10.8, mask=None):
    """
    Compute where and when daily average wind speed is above a given threshold (windy days).

    Input
    -------
    ws: xarray.DataArray ; (time,lat,lon)
        Daily wind speed magnitude.
    threshold: float / int
        Wind speed threshold (default: 10.8 m/s). Windy day threshold.
    mask: str or None (None by default)
        Path to a netCDF file containing a land-sea mask. If provided, the mask will be applied.

    Output
    -------
    wd: xarray.DataArray ; (lat,lon)
        Number of windy days.

    References
    -------
    [1]: https://xclim.readthedocs.io/en/stable/indices.html#xclim.indices.windy_days
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(ws) == "DataArray"
    ), 'The input variable "ws" is not an xarray.DataArray.'
    if not isinstance(threshold, (float, int)):
        raise TypeError('The input variable "threshold" must be a float or an integer.')
    # Check the dimensions of the input variables.
    assert (
        ws.ndim == 3
    ), 'The input variable "ws" does not have the required dimensions (time,lat,lon).'

    # Apply the land-sea mask by default.
    if mask is not None:
        ws = onshore_mask(ws, mask)
    if "level" in ws.dims:
        ws = ws.squeeze("level", drop=True)

    # Compute the number of windy days.

    wd = np.sum(ws > threshold, axis=0)

    # Add metadata to the output variable.
    attrs = {"shortname": "wd", "longname": "Number of windy days", "units": "-"}
    coords = {"lat": ws.lat, "lon": ws.lon}
    dims = ("lat", "lon")

    wd = xr.DataArray(
        wd, coords=coords, dims=dims, attrs=attrs, name=attrs["shortname"]
    )

    return wd.to_dataset()


def capacity_factor_histogram_opa(cf, working_dir, mask=None, iec_class='I'):
    """
    Compute the capacity factor histogram for a given grid.

    Input
    -------
    cf: xarray.DataArray ; (time,lat,lon)
        Wind speed magnitude at hub height.
    working_dir: string
        Directory where opa works and where the output is dumped.
    iec_class: str
        IEC wind turbine class. Options are 'I', 'II', 'III', 'S'.
    Output
    -------
    one_pass pickle file (if statistic is on the making) or netcdf (if statistic is completed).

    References
    -------
    [1]: https://earth.bsc.es/gitlab/digital-twins/de_340-2/one_pass/
    """
    # Get data from GSV

    #variable = list(cf.data_vars)[0] # cf_X
    #print(variable)

    oparequest = {
        "stat" : "histogram",
        "stat_freq": "monthly",
        "output_freq": "monthly",
        "time_step": 60,
        "variable": f"cf_{iec_class.lower()}",
        "save": True,
        "bins": 30,
        "checkpoint": True,
        "checkpoint_filepath": f'{working_dir}',
        "save_filepath": f'{working_dir}',
    }

    # Apply mask to data
    if mask is not None and mask.lower() != "none":
        print(f"Applying mask: {mask}")

        files_cf = glob.glob(f"{working_dir}/*cf_{iec_class.upper()}.nc")

        if files_cf==[]:
            print('No histogram file to be processed.')
        else:
            # find latest file
            latest_file = max(files_cf, key=os.path.getmtime)



        data = xr.open_dataset(latest_file, engine="netcdf4")
        data_attrs = data.attrs
        mask_data = xr.open_dataset(mask, engine="netcdf4").mask
        mask_data.attrs = data_attrs
        data = apply_mask(data, mask_data)
        data.attrs = data_attrs
        cf=data
    else:
        print("No mask applied.")
        
    # Run One Pass algorithm on a specific stat & variable controlled by the oparequest
    opa_stat = Opa(oparequest)
    opa_stat.compute(cf)
    
    if mask is not None and mask.lower() != "none":
        print(f"Applying reshape to original format.")
        mask_data=xr.open_dataset(mask, engine="netcdf4").mask

        files = glob.glob(f"{working_dir}/*_{iec_class.lower()}_*{oparequest['stat']}*")

        if glob.glob(f"{working_dir}/*_{iec_class.lower()}_*{oparequest['stat']}*"): # if histogram do counts and edges
            files_counts = glob.glob(f"{working_dir}/*_{iec_class.lower()}_*counts*.nc")
            files_edges = glob.glob(f"{working_dir}/*_{iec_class.lower()}_*edges*.nc")
            if files_counts==[] and files_edges==[]:
                print('No histogram file to be processed.')
            else:

                # find latest file
                latest_counts = max(files_counts, key=os.path.getmtime)
                latest_edges = max(files_edges, key=os.path.getmtime)

                # load data
                bin_counts_data = xr.open_dataset(f'{latest_counts}', engine="netcdf4")
                bin_edges_data = xr.open_dataset(f'{latest_edges}', engine="netcdf4")

                # get original shape and save
                bin_counts = reshape_final_file(bin_counts_data, mask_data)
                bin_edges = reshape_final_file(bin_edges_data, mask_data)
                bin_counts.to_netcdf(f"{latest_counts}_final")
                bin_edges.to_netcdf(f"{latest_edges}_final")
                print(f'Histograms final file saved at original shape at {latest_edges} and {latest_counts}.')

############## teporary code to speed up the opa, until opa performance improves.
#================================================================================
def apply_mask(data, mask):
    """Apply mask to an xarray dataset.

    Parameters
    ----------
    data : xarray.DataArray
        xarray data array containing the output of the gsv_interface
    mask : xarray.DataArray
        Boolean mask to apply, of the same spatial (lat lon) shape as "data" input xarray.

    Returns
    -------
    masked_data
        Masked xarray dataarray of shape (time:24, lat:1, lon:sum(mask)).
    """
    # Get grid points of the mask.
    mask_size = int(sum(mask.data.flatten()))

    # add time dimension
    mask_expanded = mask.expand_dims({"time": data.time}, axis=0)

    # Create new DataArray
    new_dummy_xarray = xr.DataArray(
        np.full(
            (24, 1, mask_size), np.nan, dtype=np.float32
        ),  # shape = (time, lat, lon)
        dims=("time", "lat", "lon"),
        coords={
            "time": data.time,  # 0..23 hours
            "lat": np.arange(1),  # 0
            "lon": np.arange(mask_size),  # 0..sum(mask)-1
        },
    )

    var_name = list(data.variables.keys())[3]
    if var_name == 'cf_i':
        data_flat = data.cf_i.data.flatten()
    elif var_name == 'cf_ii':
        data_flat = data.cf_ii.data.flatten()
    elif var_name == 'cf_iii':
        data_flat = data.cf_iii.data.flatten()
    elif var_name == 'cf_s':
        data_flat = data.cf_s.data.flatten()
    else:
        raise ValueError("Variable name not recognised.")

    # flatten the xarrays
    mask_flat = mask_expanded.data.flatten()

    if len(data_flat) != len(mask_flat):
        raise ValueError("Shapes of data and mask are not equal.")

    # select the points that have some meaning (not masked points).
    masked_data = data_flat[mask_flat == 1]

    # reduce memory usage
    del data_flat, mask_flat

    reshaped = masked_data.reshape(24, 1, mask_size)

    new_dummy_xarray.values[:] = reshaped

    masked_data = new_dummy_xarray

    return masked_data

def reshape_final_file(data, mask):
    """Reshape final output file after applying mask into original shape.

    Parameters
    ----------
    data : xarray.DataArray
        xarray data array containing the output of the t-digest flattened masked file.
    mask :
        Boolean mask to apply, of the same shape as "data" input xarray.

    Returns
    -------
    reshaped_data
        Reshaped xarray dataarray of shape (time:1, lat:mask.lat lon:mask.lon).
    """
    #mask=mask_ds.mask

    indices=np.where(mask==1)

    data_to_insert = data['None'].squeeze('lat').data  # shape (1, dimsize, N)

    print(f'data_to_insert: {data_to_insert}')

    if 'percentile' in data['None'].dims:
        dim='percentile'
        dimsize=1
    elif 'bin_count' in data['None'].dims:
        dim='bin_count'
        dimsize=30
    elif 'bin_edges' in data['None'].dims:
        dim='bin_edges'
        dimsize=31
    else:
        print('Non recognised variable name.')
    
    new_dummy_xarray = xr.DataArray(
                 np.full((1, dimsize, len(mask.lat), len(mask.lon)), np.nan, dtype=np.float32),  # shape = (time, lat, lon)
              dims=("time", dim, "lat", "lon"),
              coords={
                  "time": 1,
                  f"{dim}": np.arange(dimsize),               # 0..dimsize
                  "lat": mask.lat,                  # 0
                  "lon": mask.lon,    # 0..sum(mask)-1
              }
    )

    arr = new_dummy_xarray.data  # direct access to underlying NumPy array

    ilat=indices[0]

    ilon=indices[1]

    # vectorized assignment
    arr[:, :, ilat, ilon] = data_to_insert

    # Suppose you still have the coordinate information:
    # (same shape and dims as your new_dummy_xarray)

    da_out = xr.DataArray(
        arr,
        dims=("time", dim, "lat", "lon"),
        coords={
            "time": data['None'].time,
            f'{dim}': new_dummy_xarray[f'{dim}'],
            "lat": new_dummy_xarray.lat,
            "lon": new_dummy_xarray.lon,
        },
        name="filled_data"
    )

    # reduce memory usage
    del arr, ilon, ilat, data_to_insert

    return da_out
