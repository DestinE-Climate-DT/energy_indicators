#!/usr/bin/env python3
"""
# Destination Earth: Energy Indicators application, demand indicators
"""

# External imports
import os
import xarray as xr
import numpy as np

from .core import get_type

def cooling_degree_days(tm, tx, tn, base=22.0):
    """
    Compute the average cooling degree days. Requires daily mean, maximum and minimum temperature.
    Base temperature can be adjusted according to the region/country considered.

    Input
    -------
    tm: xarray.DataArray ; (time,lat,lon)
        Mean temperature in °C.
    tx: xarray.DataArray ; (time,lat,lon)
        Maximum temperature in °C.
    tn: xarray.DataArray ; (time,lat,lon)
        Minimum temperature in °C.
    base: float
        Base temperature (default: 22°C). Depends on the region/country considered.

    Output
    -------
    cdd: xarray.DataArray ; (time,lat,lon)
        Cooling degree days.
    cdd_acc: xarray.DataArray ; (lat,lon)
        Total accumulated cooling degree days.

    References
    -------
    [1]: https://doi.org/10.1002/joc.3959
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(tm) == "DataArray"
    ), 'The input variable "tm" is not an xarray.DataArray.'
    assert (
        get_type(tx) == "DataArray"
    ), 'The input variable "tx" is not an xarray.DataArray.'
    assert (
        get_type(tn) == "DataArray"
    ), 'The input variable "tn" is not an xarray.DataArray.'
    if not isinstance(base, (float, int)):
        raise TypeError('The input variable "base" must be a float or an integer.')
    # Check the dimensions of the input variables.
    for var in [tm, tx, tn]:
        assert (
            var.ndim == 3
        ), f"The input variable {var.name} does not have the required dimensions (time,lat,lon)."

    # Initialize the cooling degree days array.
    cdd = np.zeros_like(tm)

    # Vectorized implementation of the cooling degree days computation.
    mask1 = base >= tx
    mask2 = np.logical_and(tm <= base, base < tx)
    mask3 = np.logical_and(tn <= base, base < tm)
    mask4 = base < tn

    cdd += np.where(mask1, 0, 0)
    cdd += np.where(mask2, (tx - base) / 4, 0)
    cdd += np.where(mask3, (tx - base) / 2 - (base - tn) / 4, 0)
    cdd += np.where(mask4, tm - base, 0)

    # Compute the accumulated cooling degree days for the period of interest.
    cdd_acc = np.sum(cdd, axis=0)

    # Add metadata to the output variable.
    attrs = {"shortname": "cdd", "longname": "Cooling degree days", "units": "°C"}
    attrs_acc = {
        "shortname": "cdd_acc",
        "longname": "Total accumulated cooling degree days",
        "units": "°C",
    }
    coords = {"time": tm.time, "lat": tm.lat, "lon": tm.lon}
    coords_acc = {"lat": tm.lat, "lon": tm.lon}
    dims = ("time", "lat", "lon")
    dims_acc = ("lat", "lon")

    cdd = xr.DataArray(
        cdd, coords=coords, dims=dims, attrs=attrs, name=attrs["shortname"]
    )
    cdd_acc = xr.DataArray(
        cdd_acc,
        coords=coords_acc,
        dims=dims_acc,
        attrs=attrs_acc,
        name=attrs_acc["shortname"],
    )

    return cdd.to_dataset() , cdd_acc.to_dataset()


def heating_degree_days(tm, tx, tn, base=15.5):
    """
    Compute the average heating degree days. Requires daily mean, maximum and minimum temperature.
    Base temperature can be adjusted according to the region/country considered.

    Input
    -------
    tm: xarray.DataArray ; (time,lat,lon)
        Mean temperature in °C.
    tx: xarray.DataArray ; (time,lat,lon)
        Maximum temperature in °C.
    tn: xarray.DataArray ; (time,lat,lon)
        Minimum temperature in °C.
    base: float
        Threshold temperature (default: 15.5°C). Depends on the region/country considered.

    Output
    -------
    hdd: xarray.DataArray ; (time,lat,lon)
        Heating degree days.
    hdd_acc: xarray.DataArray ; (lat,lon)
        Total accumulated heating degree days.

    References
    -------
    [1]: https://doi.org/10.1002/joc.3959
    """
    # Check if the input parameters satisfy the required conditions.
    assert (
        get_type(tm) == "DataArray"
    ), 'The input variable "tm" is not an xarray.DataArray.'
    assert (
        get_type(tx) == "DataArray"
    ), 'The input variable "tx" is not an xarray.DataArray.'
    assert (
        get_type(tn) == "DataArray"
    ), 'The input variable "tn" is not an xarray.DataArray.'
    if not isinstance(base, (float, int)):
        raise TypeError('The input variable "base" must be a float or an integer.')
    # Check the dimensions of the input variables.
    for var in [tm, tx, tn]:
        assert (
            var.ndim == 3
        ), f"The input variable {var.name} does not have the required dimensions (time,lat,lon)."

    # Initialize the heating degree days array.
    hdd = np.zeros_like(tm)

    # Vectorized implementation of the heating degree days computation.
    mask1 = base >= tx
    mask2 = np.logical_and(tm <= base, base < tx)
    mask3 = np.logical_and(tn <= base, base < tm)
    mask4 = base < tn

    hdd += np.where(mask1, base - tm, 0)
    hdd += np.where(mask2, (base - tn) / 2 - (tx - base) / 4, 0)
    hdd += np.where(mask3, (base - tn) / 4, 0)
    hdd += np.where(mask4, 0, 0)

    # Compute the total heating degree days for the period of interest.
    hdd_acc = np.sum(hdd, axis=0)

    # Add metadata to the output variable.
    attrs = {"shortname": "hdd", "longname": "Heating degree days", "units": "°C"}
    attrs_acc = {
        "shortname": "hdd_acc",
        "longname": "Total accumulated heating degree days",
        "units": "°C",
    }
    coords = {"time": tm.time, "lat": tm.lat, "lon": tm.lon}
    coords_acc = {"lat": tm.lat, "lon": tm.lon}
    dims = ("time", "lat", "lon")
    dims_acc = ("lat", "lon")

    hdd = xr.DataArray(
        hdd, coords=coords, dims=dims, attrs=attrs, name=attrs["shortname"]
    )
    hdd_acc = xr.DataArray(
        hdd_acc,
        coords=coords_acc,
        dims=dims_acc,
        attrs=attrs_acc,
        name=attrs_acc["shortname"],
    )

    return hdd.to_dataset(), hdd_acc.to_dataset()
