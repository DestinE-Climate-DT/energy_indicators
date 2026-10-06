import xarray as xr
import numpy as np
import pandas as pd
from energy_indicators import (
    pv_pot,
    cell_temp,
    daily_cell_temp,
    daily_pv_potential,
)

from energy_indicators.core import wind_speed

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
    assert isinstance(out, xr.Dataset)
    assert "pvp" in out
    assert (out.pvp >= 0).all()


def test_cell_temp(
    dataarray_t_c: xr.DataArray,
    dataarray_avg_sdswrf: xr.DataArray,
    dataarray_10si: xr.DataArray,
):
    # prepare inputs in (time, lat, lon)
    t_c = dataarray_t_c.sel({"variable": "t_c"}, drop=True).transpose("time", "lat", "lon")
    g   = dataarray_avg_sdswrf.sel({"variable": "avg_sdswrf"}, drop=True).transpose("time", "lat", "lon")
    ws  = dataarray_10si.sel({"variable": "10si"}, drop=True).transpose("time", "lat", "lon")

    out = cell_temp(t_c, g, ws)

    # minimal checks (same spirit as test_wind.py)
    assert isinstance(out, xr.Dataset)
    assert "cell_temp" in out
    assert out.cell_temp.all()


def test_daily_cell_temp(
    dataarray_t_c: xr.DataArray,
    dataarray_avg_sdswrf: xr.DataArray,
    dataarray_10si: xr.DataArray,
):
    # prepare inputs in (time, lat, lon)
    t_c = dataarray_t_c.sel({"variable": "t_c"}, drop=True).transpose("time", "lat", "lon")
    g   = dataarray_avg_sdswrf.sel({"variable": "avg_sdswrf"}, drop=True).transpose("time", "lat", "lon")
    ws  = dataarray_10si.sel({"variable": "10si"}, drop=True).transpose("time", "lat", "lon")

    ct = cell_temp(t_c, g, ws)

    out = daily_cell_temp(ct)

    # minimal checks (same spirit as test_wind.py)
    assert isinstance(out, xr.Dataset)
    assert "daily_avg_cell_temp" in out
    assert "daily_max_cell_temp" in out
    assert (out.daily_avg_cell_temp <= out.daily_max_cell_temp).all()


def test_daily_pv_potential(
    dataarray_t_c: xr.DataArray,
    dataarray_avg_sdswrf: xr.DataArray,
    dataarray_10si: xr.DataArray,
):
    # prepare inputs in (time, lat, lon)
    t_c = dataarray_t_c.sel({"variable": "t_c"}, drop=True).transpose("time", "lat", "lon")
    g   = dataarray_avg_sdswrf.sel({"variable": "avg_sdswrf"}, drop=True).transpose("time", "lat", "lon")
    ws  = dataarray_10si.sel({"variable": "10si"}, drop=True).transpose("time", "lat", "lon")

    pvp = pv_pot(t_c, g, ws)

    out = daily_pv_potential(pvp)

    # minimal checks (same spirit as test_wind.py)
    assert isinstance(out, xr.Dataset)
    assert "pvp" in out
    assert (out.pvp >= 0).all()

    # daily mean must match resampling the hourly pvp directly, regardless
    # of how many timestamps the fixture puts in each calendar day
    expected = pvp["pvp"].resample(time="1D").mean(dim="time", skipna=True)
    xr.testing.assert_allclose(out["pvp"], expected)
