#!/usr/bin/env python3
"""
| Destination Earth: Energy Indicators application

Solar indicators. Currently provides PV potential; further solar
indicators are planned.
"""

# External libraries
import xarray as xr

# Internal libraries
from .core import get_type


# Define PV potential calculation function


def pv_pot(t2c, g, ws):
    """
    Compute the PV potential (pvp) based on hourly solar radiation, temperature, and wind speed.

    Input
    -------
    t2c: xarray.DataArray ; (time,lat,lon)
        2m air temperature (tas) in °C.
    g: xarray.DataArray ; (time,lat,lon)
        Surface solar radiation downwards (rsds/avg_sdswrf) in W/m².
    ws: xarray.DataArray ; (time,lat,lon)
        Surface wind speed (sfcWind) in m/s.

    If the temperature data is in Kelvin, use ``convert_temperature`` from
    ``core`` to convert to °C. If radiation data is in J/m², use
    ``convert_radiation`` from ``core`` to convert to W/m². ``ws`` can be
    computed using ``wind_speed`` from ``core`` given u10/v10 components.

    Output
    -------
    pvp: xarray.Dataset ; (time,lat,lon)
        PV potential, at the same temporal frequency as the inputs (hourly
        if hourly inputs are given). Use ``daily_pv_potential`` to aggregate
        to a daily mean.

    References
    -----------
    [1]: https://iopscience.iop.org/article/10.1088/1748-9326/ad8c68/meta
    [2]: https://doi.org/10.1038/ncomms10014
    """
    # Check if the input parameters satisfy the required conditions.
    assert get_type(t2c) == "DataArray", (
        'The input variable "t2c" is not an xarray.DataArray.'
    )
    assert get_type(g) == "DataArray", (
        'The input variable "rsds/avg_sdswrf" is not an xarray.DataArray.'
    )
    assert get_type(ws) == "DataArray", (
        'The input variable "ws10/10si/sfcWind" is not an xarray.DataArray.'
    )

    # Check input dimensions
    for var, name in zip([t2c, g, ws], ["t2c", "g", "ws"]):
        assert var.ndim == 3, (
            f"The input variable {name} does not have \
            the required dimensions (time, lat, lon)."
        )

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
        "units": "[MW/MW]",
    }
    coords = {"time": g.time, "lat": g.lat, "lon": g.lon}
    dims = ("time", "lat", "lon")

    # Convert to xarray DataArray
    pvp = xr.DataArray(
        pvpot, dims=dims, coords=coords, attrs=attrs, name=attrs["shortname"]
    )

    return pvp.to_dataset()


# Define daily PV potential aggregation function


def daily_pv_potential(pvp):
    """
    Compute daily mean PV potential from hourly PV potential.

    PVP is nonlinear in ``g`` (quadratic) and in the ``g * t2c`` / ``g * ws``
    cross-terms (see ``pv_pot``), so the daily mean must be computed by
    averaging hourly PVP, it is not equivalent to computing PVP from
    daily-mean temperature, radiation and wind speed.

    Input
    -------
    pvp: xarray.Dataset or xarray.DataArray
        Output from the pv_pot function (hourly), or a DataArray containing
        hourly PV potential.

    Output
    -------
    daily_pvp: xarray.Dataset ; (time, lat, lon)
        Dataset containing daily mean PV potential.

        Variables:
        - pvp

    References
    -----------
    [1]: https://iopscience.iop.org/article/10.1088/1748-9326/ad8c68/meta
    [2]: https://doi.org/10.1038/ncomms10014
    """
    # If input is Dataset from pv_pot function, select the pvp variable
    if isinstance(pvp, xr.Dataset):
        pvp = pvp["pvp"]

    # Check if the input parameter satisfies the required condition.
    assert get_type(pvp) == "DataArray", (
        'The input variable "pvp" is not an xarray.DataArray.'
    )

    # Check input dimensions
    assert pvp.ndim == 3, (
        'The input variable "pvp" does not have '
        "the required dimensions (time, lat, lon)."
    )

    # Compute daily mean PV potential
    daily_pvp = pvp.resample(time="1D").mean(dim="time", skipna=True)

    # Assign attributes and name
    attrs = {
        "shortname": "pvp",
        "longname": "Daily mean PV Potential",
        "units": "[MW/MW]",
    }
    daily_pvp.attrs = attrs
    daily_pvp.name = attrs["shortname"]

    return daily_pvp.to_dataset()


# Define cell temperature calculation function


def cell_temp(t2c, g, ws):
    """
    Compute the cell temperature (cell_temp) based on hourly 
    solar radiation, temperature, and wind speed.

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
    cell_temp: xarray.DataArray ; (time, lat, lon)
        Computed cell temperature values at any temporal frequency, 
        depending on the input parameters.

    References
    -------
    [1]: https://iopscience.iop.org/article/10.1088/1748-9326/ad8c68/meta
    [2]: https://iopscience.iop.org/article/10.1088/1748-9326/ac2a64/meta
    """
    # Check if the input parameters satisfy the required conditions.
    assert get_type(t2c) == "DataArray", (
        'The input variable "t2c" is not an xarray.DataArray.'
    )
    assert get_type(g) == "DataArray", (
        'The input variable "rsds/avg_sdswrf" is not an xarray.DataArray.'
    )
    assert get_type(ws) == "DataArray", (
        'The input variable "ws10/10si/sfcWind" is not an xarray.DataArray.'
    )

    # Check input dimensions
    for var, name in zip([t2c, g, ws], ["t2c", "g", "ws"]):
        assert var.ndim == 3, (
            f"The input variable {name} does not have "
            "the required dimensions (time, lat, lon)."
        )

    # Define cell temperature coefficients
    c1 = 4.3  # °C
    c2 = 0.943  # unitless
    c3 = 0.028  # °C m^2 W^-1
    c4 = -1.528  # °C s m^-1

    # Compute cell temperature
    celltemp = c1 + c2 * t2c + c3 * g + c4 * ws

    # Assign attributes correctly
    attrs = {
        "shortname": "cell_temp",
        "longname": "Solar cell temperature",
        "units": "°C",
    }

    coords = {"time": g.time, "lat": g.lat, "lon": g.lon}
    dims = ("time", "lat", "lon")

    # Convert to xarray DataArray
    celltemp_da = xr.DataArray(
        celltemp,
        dims=dims,
        coords=coords,
        attrs=attrs,
        name=attrs["shortname"],
    )

    return celltemp_da.to_dataset()


# Define daily cell temperature indicators function


def daily_cell_temp(celltemp):
    """
    Compute daily average and daily maximum solar cell temperature.

    Input
    -------
    celltemp: xarray.Dataset or xarray.DataArray
        Output from the cell_temp function or a DataArray containing solar cell
        temperature in degree C.

    Output
    -------
    daily_cell_temp: xarray.Dataset ; (time, lat, lon)
        Dataset containing daily average and daily maximum solar cell temperature.

        Variables:
        - daily_avg_cell_temp
        - daily_max_cell_temp
    References
    -------
    [1] https://iopscience.iop.org/article/10.1088/1748-9326/ac2a64/meta
    [2] https://iopscience.iop.org/article/10.1088/1748-9326/ad8c68/meta
    """
    # If input is Dataset from cell_temp function, select the cell_temp variable
    if isinstance(celltemp, xr.Dataset):
        celltemp = celltemp["cell_temp"]

    # Check if the input parameter satisfies the required condition.
    assert get_type(celltemp) == "DataArray", (
        'The input variable "celltemp" is not an xarray.DataArray.'
    )

    # Check input dimensions
    assert celltemp.ndim == 3, (
        'The input variable "celltemp" does not have '
        "the required dimensions (time, lat, lon)."
    )

    # Compute daily average and daily maximum cell temperature
    daily_avg_celltemp = celltemp.resample(time="1D").mean(
        dim="time", skipna=True
    )
    daily_max_celltemp = celltemp.resample(time="1D").max(
        dim="time", skipna=True
    )

    # Assign attributes for daily average cell temperature
    attrs_avg = {
        "shortname": "daily_avg_cell_temp",
        "longname": "Daily average solar cell temperature",
        "units": "°C",
    }

    # Assign attributes for daily maximum cell temperature
    attrs_max = {
        "shortname": "daily_max_cell_temp",
        "longname": "Daily maximum solar cell temperature",
        "units": "°C",
    }

    # Assign attributes and names
    daily_avg_celltemp.attrs = attrs_avg
    daily_avg_celltemp.name = attrs_avg["shortname"]

    daily_max_celltemp.attrs = attrs_max
    daily_max_celltemp.name = attrs_max["shortname"]

    # Convert both DataArrays to Dataset and merge
    daily_ct = xr.merge(
        [
            daily_avg_celltemp.to_dataset(),
            daily_max_celltemp.to_dataset(),
        ]
    )

    # Assign global Dataset attributes
    daily_ct.attrs = {
        "shortname": "daily_cell_temp",
        "longname": "Daily solar cell temperature indicators",
        "units": "°C",
    }

    return daily_ct

# Development of solar energy indicators for the Energy Indicators application.

# Ideas for future development.
# def effective_radiation_days(rsds):
#    return None


# def cloud_days(clt):
#    return None


# def clear_sky_days(clt):
#    return None
