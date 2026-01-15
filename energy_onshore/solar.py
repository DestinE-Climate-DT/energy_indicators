#!/usr/bin/env python3
"""
# Destination Earth: Energy Onshore application
# Author: Aleks Lacima
# Version: 0.5.0
"""

# Load libraries
import xarray as xr
import numpy as np

from .core import get_type

# Development of solar energy indicators for the Energy Onshore application.


def solar_capacity_factor_daily(t2c, rsds):
    """
    Compute the capacity factor of a PV solar panel at daily frequency.

    Input
    -------
    t2c: xarray.DataArray ; (time,lat,lon)
        Daily temperature at 2m in °C.
    rsds: xarray.DataArray ; (time,lat,lon)
        Daily surface solar radiation donwnwards in W m^(-2).

    Output
    -------
    cf_daily: xarray.DataArray ; (time,lat,lon)
        Daily capacity factor.

    References
    -------
    [1]: https://doi.org/10.1016/j.renene.2015.10.006
    [2]: https://doi.org/10.1038/ncomms10014
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(t2c) == "DataArray"
    ), 'The input variable "t2c" is not an xarray.DataArray.'
    assert (
        get_type(rsds) == "DataArray"
    ), 'The input variable "rsds" is not an xarray.DataArray.'
    # Check the dimensions of the input variables.
    for var in [rsds, t2c]:
        assert (
            var.ndim == 3
        ), f'The input variable {var.name} does not have the required dimensions (time,lat,lon).'

    # Compute the PV module temperature.
    t_noct = 48.0  # Nominal operating cell temperature (°C)
    t_stc = 25.0  # Standard test conditions (°C)
    t0 = 20.0  # Reference temperature (°C)
    g0 = 800.0  # Reference irradiance (W m^(-2))
    g_stc = 1000.0  # Standard test conditions irradiance (W m^(-2))

    t_mod = t2c + (t_noct - t0) * (rsds / g0)
    dt_mod = t_mod - t_stc

    # Compute the capacity factor.
    alfa = 0.00042  # K^(-1)
    beta = -0.00046  # K^(-1)
    c1 = 0.033
    c2 = -0.0092

    cf_daily = (1 + alfa * dt_mod) * (
        1 + c1 * np.log(rsds / g_stc) + c2 * (np.log(rsds / g_stc)) ** 2 + beta * dt_mod
    )

    # Add metadata to the output variable.
    attrs = {"shortname": "cf", "longname": "Capacity factor", "units": "%"}
    coords = {"time": rsds.time, "lat": rsds.lat, "lon": rsds.lon}
    dims = ("time", "lat", "lon")

    cf_daily = xr.DataArray(
        cf_daily, coords=coords, dims=dims, attrs=attrs, name=attrs["shortname"]
    )

    return cf_daily


def annual_energy_production_daily(capacity_factor, rated_power, num_panels=1):
    """
    Compute the annual energy production of a PV solar panel.

    Input
    -------
    capacity_factor_series: xarray.DataArray ; (time)
        Capacity factor time series for a year.
    rated_power: float / int
        Rated power of the solar panel in kW.
    num_turbines: int
        Number of solar panels (default: 1).

    Output
    -------
    aep: xarray.DataArray ; (time)
        Annual energy production in kWh.
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(capacity_factor) == "DataArray"
    ), 'The input variable "capacity_factor" is not an xarray.DataArray.'
    if not isinstance(rated_power, (float, int)):
        raise TypeError(
            'The input variable "rated_power" must be a float or an integer.'
        )
    if not isinstance(num_panels, int):
        raise TypeError('The input variable "num_panels" must be an integer.')
    # Check the dimensions of the input variables.
    assert (
        capacity_factor.ndim == 1
    ), 'The input variable "capacity_factor" does not have the required dimensions (time).'

    # Compute the annual energy production.
    nhours = 24
    aep_daily = capacity_factor * rated_power * num_panels * nhours

    # Add metadata to the output variable.
    attrs = {"shortname": "aep", "longname": "Annual energy production", "units": "KWh"}
    coords = {"time": capacity_factor.time}
    dims = "time"

    aep_daily = xr.DataArray(
        aep_daily, coords=coords, dims=dims, attrs=attrs, name=attrs["shortname"]
    )

    return aep_daily


# Pending to be completed.
#def solar_capacity_factor_hourly(rsds, t, elevation):
#    """
#    Compute the capacity factor of a PV solar panel at hourly scale (taking into account the \
#            solar elevation angle).
#
#    Input
#    -------
#    rsds: xarray.DataArray ; (time,lat,lon)
#        Surface solar radiation donwnwards in W m^(-2).
#    t: xarray.DataArray ; (time,lat,lon)
#        Temperature in °C.
#
#    Output
#    -------
#    cf: xarray.DataArray ; (time,lat,lon)
#        Capacity factor.
#
#    References
#    -------
#    [1]:
#    """
#
#    # Check if the input parameters satisfy the required conditions.
#    assert (
#        get_type(rsds) == "DataArray"
#    ), 'The input variable "rsds" is not an xarray.DataArray.'
#    assert (
#        get_type(t) == "DataArray"
#    ), 'The input variable "t" is not an xarray.DataArray.'
#    # Check the dimensions of the input variables.
#    for var in [rsds, t]:
#        assert (
#            var.ndim == 3
#        ), f'The input variable {var.name} does not have the required dimensions (time,lat,lon).'
#
#    # Compute the capacity factor.
#
#    # Add metadata to the output variable.
#    attrs = {"shortname": "cf", "longname": "Capacity factor", "units": "%"}
#    coords = {"time": rsds.time, "lat": rsds.lat, "lon": rsds.lon}
#    dims = ("time", "lat", "lon")
#
#    cf = xr.DataArray(
#        cf, coords=coords, dims=dims, attrs=attrs, name=attrs["shortname"]
#    )
#
#    return cf


# Ideas for future development.
#def effective_radiation_days(rsds):
#    return None


#def cloud_days(clt):
#    return None


#def clear_sky_days(clt):
#    return None
