#!/usr/bin/env python3
"""
# Destination Earth: Energy Indicators application
# Authors: Aleksander Lacma-Nadolnik, Francesc Roura-Adserias, Sushovan Ghosh
# Version: 1.1.x
"""

# External libraries
import xarray as xr

# Internal libraries
from .core import get_type


# Define PV potential calculation function

def pv_pot(t2c, g, ws):
    """
    Compute the PV potential (PV_pot) based on hourly solar radiation, temperature, and wind speed.

    Input
    -------
    t2c: a DataArray, containing information of 2m air temperature (tas) in degree C.
    g:   a DataArray, containing surface solar radiation downwards (rsds/avg_sdswrf) in W/m2.
    ws:  a DataArray, containing surface wind speed (sfcWind) in m/s.

    If the temperature data is in Kelvein, use convert_temperature function from core \
    to convert in Degree C.
    If radiation data is in J/m2, use convert_radiation from core to transform in W/m2

    ws could be calculated using surface_wind function from core by using u10 and v10 data

    Output
    -------
    pv_pot: xarray.DataArray ; (time, lat, lon)
        Computed PV potential values at any temporal frequency depending on the input parameters.

    References
    -------
    [1]: https://iopscience.iop.org/article/10.1088/1748-9326/ad8c68/meta
    [2]: https://doi.org/10.1038/ncomms10014
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(t2c) == "DataArray"
    ), 'The input variable "t2c" is not an xarray.DataArray.'
    assert (
        get_type(g) == "DataArray"
    ), 'The input variable "rsds/avg_sdswrf" is not an xarray.DataArray.'
    assert (
        get_type(ws) == "DataArray"
    ), 'The input variable "ws10/10si/sfcWind" is not an xarray.DataArray.'


    # Check input dimensions
    for var, name in zip([t2c, g, ws], ["t2c", "g", "ws"]):
        assert var.ndim == 3, f"The input variable {name} does not have \
            the required dimensions (time, lat, lon)."

    # Define PV potential coefficients
    alpha1 = 1.1035e-3  # (W m^-2)^-1
    alpha2 = 1.4e-7  # (W m^-2)^-2
    alpha3 = -4.715e-6  # (W°C m^-2)^-1
    alpha4 = 7.64e-6  # (W ms)^-1

    # Compute PV potential (PV_pot)
    pvpot = alpha1 * g + alpha2 * g**2 + alpha3 * g * t2c + alpha4 * g * ws

    # Assign attributes correctly
    attrs = {
        "shortname": "pvp",
        "longname": "PV Potential",
        "units": "dimensionless",
        "description": "Computed PV potential using surface solar radiation,\
            temperature, and wind speed.",
        "Reference": "https://doi.org/10.1038/ncomms10014 ,\
            https://iopscience.iop.org/article/10.1088/1748-9326/ad8c68/meta",
        "Contact": "BSC Energy Use Case Team",
    }
    coords = {"time": g.time, "lat": g.lat, "lon": g.lon}
    dims = ("time", "lat", "lon")

    # Convert to xarray DataArray
    pvp = xr.DataArray(pvpot, dims=dims, coords=coords, attrs=attrs, name=attrs["shortname"])

    return pvp


# Development of solar energy indicators for the Energy Indicators application.

# Ideas for future development.
# def effective_radiation_days(rsds):
#    return None


# def cloud_days(clt):
#    return None


# def clear_sky_days(clt):
#    return None