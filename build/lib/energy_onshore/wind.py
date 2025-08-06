#!/usr/bin/env python3
"""
# Destination Earth: Energy Indicators application
"""

# External imports
import os
import xarray as xr
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from one_pass.opa import Opa


# Load libraries
from .core import get_type, select_region, wind_speed, select_point

# Development of wind energy indicators for the Energy Onshore application.


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
    #    from .core import select_region

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


def wind_power_density(ws, air_density=1.225):
    """
    Compute the wind power density.

    Input
    -------
    ws: xarray.DataArray ; (time,lat,lon)
        Wind speed magnitude.
    air_density: float or xarray.DataArray ; (time,lat,lon)
        Air density. Default value is 1.225 kg m^(-3).

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
    # from .core import wind_speed

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
    Compute the estimated power output of a wind turbine.

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
    # from scipy.optimize import curve_fit

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

    # Define a parametrized Weibull Cumulative Distribution Function to fit the power curve.
    def weibull_distribution(x, alfa, k):
        f = rated_power - rated_power * np.exp(-((x * alfa) ** k))
        return f

    x_data = power_curve_data.wind_speed[:]
    y_data = power_curve_data.power[:]

    # Fit the manufacture power curve to the Weibull Cumulative Distribution Function.
    popt, _ = curve_fit(weibull_distribution, x_data, y_data, p0=(0.1, 1))

    # Extract the optimized parameters
    alfa_opt, k_opt = popt

    # Evaluate power output at given wind speeds using the piecewise definition.
    conditions = [
        (ws < cut_in_speed),
        (ws >= cut_in_speed) & (ws <= rated_speed),
        (ws > rated_speed) & (ws <= cut_out_speed),
        (ws > cut_out_speed),
    ]

    functions = [
        0,
        lambda x: weibull_distribution(x, alfa_opt, k_opt),
        rated_power,
        0,
    ]

    # A piecewise function is used to obtain the power curve.
    power_out = np.piecewise(ws, conditions, functions)

    # Store the power curve and the wind values in an xarray.DataArray.
    attrs = {"shortname": "po", "longname": "Power output", "units": "kW"}
    coords = {"wind_speed": ws}
    dims = "wind_speed"

    power_out = xr.DataArray(
        power_out, coords=coords, dims=dims, attrs=attrs, name=attrs["shortname"]
    )

    return power_out


def capacity_factor(ws, iec_class):
    """
    Compute the capacity factor of a wind turbine.

    Input
    -------
    ws: xarray.DataArray ; (time,lat,lon)
        Wind speed magnitude at hub height.
    iec_class: str
        IEC wind turbine class. Options are 'I','I/II', 'II', 'II/III', 'III', 'S'.

    Output
    -------
    cf: xarray.DataArray ; (time,lat,lon)
        Capacity factor.

    References
    -------
    [1]: https://doi.org/10.1016/j.renene.2019.04.135
    [2]: https://docs.scipy.org/doc/scipy/reference/generated/\
            scipy.spatial.KDTree.html#scipy.spatial.KDTree

    """
    # from scipy.spatial import KDTree

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

    class_i = {
        "turbine_model": "Enercon E70",
        "rotor_diameter": 71,
        "rated_power": 2300,
        "hub_height": 85,
        "cut_in_speed": 2.0,
        "rated_speed": 15.5,
        "cut_out_speed": 25.0,
    }
    class_i_ii = {
        "turbine_model": "Gamesa G80",
        "rotor_diameter": 80,
        "rated_power": 2000,
        "hub_height": 80,
        "cut_in_speed": 3.5,
        "rated_speed": 15.0,
        "cut_out_speed": 25.0,
    }
    class_ii = {
        "turbine_model": "Gamesa G87",
        "rotor_diameter": 87,
        "rated_power": 2000,
        "hub_height": 83.5,
        "cut_in_speed": 3.0,
        "rated_speed": 14.0,
        "cut_out_speed": 25.0,
    }
    class_ii_iii = {
        "turbine_model": "Vestas V100",
        "rotor_diameter": 100,
        "rated_power": 2000,
        "hub_height": 100,
        "cut_in_speed": 3.5,
        "rated_speed": 12.0,
        "cut_out_speed": 20.0,
    }
    class_iii = {
        "turbine_model": "Vestas V110",
        "rotor_diameter": 110,
        "rated_power": 2000,
        "hub_height": 100,
        "cut_in_speed": 4.0,
        "rated_speed": 12.0,
        "cut_out_speed": 20.0,
    }
    class_s = {
        "turbine_model": "Vestas V164",
        "rotor_diameter": 164,
        "rated_power": 9500,
        "hub_height": 105,
        "cut_in_speed": 3.5,
        "rated_speed": 14.0,
        "cut_out_speed": 25.0,
    }

    turbine_class = [class_i, class_i_ii, class_ii, class_ii_iii, class_iii, class_s]

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

    # Add metadata to the output variable.
    attrs = {"shortname": "cf", "longname": "Capacity factor", "units": "-"}
    coords = {"time": ws.time, "lat": ws.lat, "lon": ws.lon}
    dims = ("time", "lat", "lon")

    cf = xr.DataArray(
        cf, coords=coords, dims=dims, attrs=attrs, name=attrs["shortname"]
    )

    return cf.to_dataset()

# warning: deprecated. cf histograms are computed via the OPA implementation under capacity_factor_histogram_opa()
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
    [1]: https://docs.scipy.org/doc/scipy/reference/generated/\
            scipy.spatial.KDTree.html#scipy.spatial.KDTree
    [2]: https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html
    [3]: https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
    """
    # from scipy.spatial import KDTree

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

    class_i = {
        "turbine_model": "Enercon E70",
        "rotor_diameter": 71,
        "rated_power": 2300,
        "hub_height": 85,
        "cut_in_speed": 2.0,
        "rated_speed": 15.5,
        "cut_out_speed": 25.0,
    }
    class_i_ii = {
        "turbine_model": "Gamesa G80",
        "rotor_diameter": 80,
        "rated_power": 2000,
        "hub_height": 80,
        "cut_in_speed": 3.5,
        "rated_speed": 15.0,
        "cut_out_speed": 25.0,
    }
    class_ii = {
        "turbine_model": "Gamesa G87",
        "rotor_diameter": 87,
        "rated_power": 2000,
        "hub_height": 83.5,
        "cut_in_speed": 3.0,
        "rated_speed": 14.0,
        "cut_out_speed": 25.0,
    }
    class_ii_iii = {
        "turbine_model": "Vestas V100",
        "rotor_diameter": 100,
        "rated_power": 2000,
        "hub_height": 100,
        "cut_in_speed": 3.5,
        "rated_speed": 12.0,
        "cut_out_speed": 20.0,
    }
    class_iii = {
        "turbine_model": "Vestas V110",
        "rotor_diameter": 110,
        "rated_power": 2000,
        "hub_height": 100,
        "cut_in_speed": 4.0,
        "rated_speed": 12.0,
        "cut_out_speed": 20.0,
    }
    class_s = {
        "turbine_model": "Vestas V164",
        "rotor_diameter": 164,
        "rated_power": 9500,
        "hub_height": 105,
        "cut_in_speed": 3.5,
        "rated_speed": 14.0,
        "cut_out_speed": 25.0,
    }

    turbine_class = [class_i, class_i_ii, class_ii, class_ii_iii, class_iii, class_s]

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
    def compute_histogram(x):
        counts, bin_edges = np.histogram(x, bins=bins)
        return counts, bin_edges

    counts, bin_edges = np.apply_along_axis(compute_histogram, axis=0, arr=cf)

    # Add metadata to the output variable.
    attrs = {"shortname": "counts", "longname": "Histogram counts", "units": "-"}
    attrs_bin = {"shortname": "bin_edges", "longname": "Bin edges", "units": "-"}
    coords = {"lat": ws.lat, "lon": ws.lon}
    dims = ("lat", "lon")

    counts = xr.DataArray(
        counts, coords=coords, dims=dims, attrs=attrs, name=attrs["shortname"]
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
    [1]: https://docs.scipy.org/doc/scipy/reference/\
            generated/scipy.spatial.KDTree.html#scipy.spatial.KDTree
    [2]: https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
    """
    #    from .core import select_point
    #    from scipy.spatial import KDTree

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

    class_i = {
        "turbine_model": "Enercon E70",
        "rotor_diameter": 71,
        "rated_power": 2300,
        "hub_height": 85,
        "cut_in_speed": 2.0,
        "rated_speed": 15.5,
        "cut_out_speed": 25.0,
    }
    class_i_ii = {
        "turbine_model": "Gamesa G80",
        "rotor_diameter": 80,
        "rated_power": 2000,
        "hub_height": 80,
        "cut_in_speed": 3.5,
        "rated_speed": 15.0,
        "cut_out_speed": 25.0,
    }
    class_ii = {
        "turbine_model": "Gamesa G87",
        "rotor_diameter": 87,
        "rated_power": 2000,
        "hub_height": 83.5,
        "cut_in_speed": 3.0,
        "rated_speed": 14.0,
        "cut_out_speed": 25.0,
    }
    class_ii_iii = {
        "turbine_model": "Vestas V100",
        "rotor_diameter": 100,
        "rated_power": 2000,
        "hub_height": 100,
        "cut_in_speed": 3.5,
        "rated_speed": 12.0,
        "cut_out_speed": 20.0,
    }
    class_iii = {
        "turbine_model": "Vestas V110",
        "rotor_diameter": 110,
        "rated_power": 2000,
        "hub_height": 100,
        "cut_in_speed": 4.0,
        "rated_speed": 12.0,
        "cut_out_speed": 20.0,
    }
    class_s = {
        "turbine_model": "Vestas V164",
        "rotor_diameter": 164,
        "rated_power": 9500,
        "hub_height": 105,
        "cut_in_speed": 3.5,
        "rated_speed": 14.0,
        "cut_out_speed": 25.0,
    }

    turbine_class = [class_i, class_i_ii, class_ii, class_ii_iii, class_iii, class_s]

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
        counts, coords=coords, dims=dims, attrs=attrs, name=attrs["shortname"]
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
    def compute_histogram(x):
        counts, bin_edges = np.histogram(x, bins=bins)
        return counts, bin_edges

    counts, bin_edges = np.apply_along_axis(compute_histogram, axis=0, arr=ws)

    # Add metadata to the output variable.
    attrs = {"shortname": "counts", "longname": "Histogram counts", "units": "-"}
    attrs_bin = {"shortname": "bin_edges", "longname": "Bin edges", "units": "-"}
    coords = {"lat": ws.lat, "lon": ws.lon}
    dims = ("lat", "lon")

    counts = xr.DataArray(
        counts, coords=coords, dims=dims, attrs=attrs, name=attrs["shortname"]
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
    #    from .core import select_point

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
    coords_bin = {"bins": np.arange(bins + 1)}
    dims = "bins"

    counts = xr.DataArray(
        counts, coords=coords, dims=dims, attrs=attrs, name=attrs["shortname"]
    )
    bin_edges = xr.DataArray(
        bin_edges,
        coords=coords_bin,
        dims=dims,
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


def high_wind_events(ws, threshold=25.0):
    """
    Compute where and when wind speed exceeds a given threshold (cut-out speed).

    Input
    -------
    ws: xarray.DataArray ; (time,lat,lon)
        Wind speed magnitude at hub height.
    threshold: float / int
        Wind speed threshold (default: 25.0 m/s). Cut-out speed of the wind turbine.

    Output
    -------
    hwe: xarray.DataArray ; (lat,lon)
        Number of high wind events.
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


def low_wind_events(ws, threshold=3.0):
    """
    Compute where and when wind speed is below a given threshold (cut-in speed).

    Input
    -------
    ws: xarray.DataArray ; (time,lat,lon)
        Wind speed magnitude at hub height.
    threshold: float / int
        Wind speed threshold (default: 3.0 m/s). Cut-in speed of the wind turbine.

    Output
    -------
    lwe: xarray.DataArray ; (lat,lon)
        Number of low wind events.
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

def capacity_factor_histogram_opa(cf,working_dir):
    """
    Compute the capacity factor histogram for a given grid.

    Input
    -------
    cf: xarray.DataArray ; (time,lat,lon)
        Wind speed magnitude at hub height.
    working_dir: string
        Directory where opa works and where the output is dumped.
    Output
    -------
    one_pass pickle file (if statistic is on the making) or netcdf (if statistic is completed).

    References
    -------
    [1]: https://earth.bsc.es/gitlab/digital-twins/de_340-2/one_pass/
    """
    oparequest = {
        "stat" : "histogram",
        "stat_freq": "weekly",
        "output_freq": "weekly",
        "time_step": 60,
        "variable": "cf",
        "save": True,
        "bins": 13,
        "checkpoint": True,
        "checkpoint_filepath": f'{working_dir}',
        "save_filepath": f'{working_dir}',
    }
    # Get data from gsv
    data = cf

    # Run One Pass algorithm on a specific stat & variable controlled by the oparequest
    opa_stat = Opa(oparequest)
    opa_stat.compute(data)

