#!/usr/bin/env python3
"""
# Destination Earth: Energy Indicators application
# Author: Aleksander Lacma-Nadolnik, Francesc Roura-Adserias, Sushovan Ghosh (Responsible: Sushovan)
# Version: 1.1.x
"""

# External libraries
import xarray as xr
import numpy as np

# Internal libraries
from .core import get_type

def onshore_mask(variable_data, mask_file):
    """
    Apply a geographical onshore mask to a given xarray DataArray using a provided mask file.

    Input
    -------
    variable_data: xarray.DataArray
        The input variable to be masked (must have lat and lon dimensions).

    mask_file: str
        Path to the NetCDF file containing the onshore mask (1 for land, 0 for ocean).

    Output
    -------
    masked_data: xarray.DataArray
        The masked variable with NaNs over ocean or excluded areas.

    Notes
    -------
    The mask is interpolated to the variable grid using nearest neighbor interpolation.
    The function assumes longitudes in the input variable may need conversion from [0, 360] to [-180, 180].

    Contact
    -------
    BSC Energy Use Case Team (energy-destine@bsc.es) Responsible: Sushovan
    """
    # Check if the input is an xarray DataArray
    assert get_type(variable_data) == "DataArray", 'The input variable is not an xarray.DataArray.'

    # Extract variable metadata
    var_name = variable_data.name if variable_data.name is not None else "unknown_variable"
    var_longname = variable_data.attrs.get('long_name', var_name)
    var_units = variable_data.attrs.get('units', '')

    # Ensure longitudes are in -180 to 180 format
    variable_data = variable_data.assign_coords(
        lon=((variable_data.lon + 180) % 360) - 180
    ).sortby('lon')

    # Load mask file
    mask_ds = xr.open_dataset(mask_file)
    mask_var = list(mask_ds.data_vars)[0]
    mask = mask_ds[mask_var]

    # Convert mask: 0 to NaN to mask excluded areas (e.g., oceans)
    mask = mask.where(mask != 0)

    # Interpolate mask to the variable grid
    mask_interp =  mask.interp(
        lat=variable_data.lat,
        lon=variable_data.lon,
        method="nearest",
        kwargs={"fill_value": np.nan}
    )

    # Optional: tighten the mask if needed (to exclude buffer zones)
    mask_interp = mask_interp.where(mask_interp > 0.5)

    # Apply the mask
    masked_data = variable_data.where(~np.isnan(mask_interp))

    return masked_data

