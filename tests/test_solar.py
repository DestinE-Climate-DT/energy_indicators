import xarray as xr
import numpy as np
import pandas as pd
from energy_onshore import (
    pv_pot,
)

from energy_onshore.core import wind_speed

"""Tests that correspond to solar.py"""

def test_pv_pot(
    dataarray_t_c: xr.DataArray,
    dataarray_avg_sdswrf: xr.DataArray,
    dataarray_10si: xr.DataArray,
):
    # prepare inputs in (time, lat, lon)
    t_c = dataarray_t_c.sel({"variable": "t_c"}, drop=True).transpose("time", "lat", "lon")
    g   = dataarray_avg_sdswrf.sel({"variable": "avg_sdswrf"}, drop=True).transpose("time", "lat", "lon")
    ws  = dataarray_10si.sel({"variable": "10si"}, drop=True).transpose("time", "lat", "lon")

    out = pv_pot(t_c, g, ws)

    # minimal checks (same spirit as test_wind.py)
    assert isinstance(out, xr.DataArray)
    assert out.dims == ("time", "lat", "lon")
    assert out.name == "pvp"
    assert out.attrs.get("units") == "dimensionless"
    assert out.all()
