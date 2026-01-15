import xarray as xr
import numpy as np
import pandas as pd

from energy_onshore import (
    wind_speed_anomalies,
    wind_power_density,
    wind_speed_at_height,
    power_output,
    capacity_factor,
    capacity_factor_histogram_1d,
    capacity_factor_histogram_opa,
    wind_speed_histogram_1d,
    annual_energy_production_wind,
    high_wind_events,
    low_wind_events,
    calm_days,
    windy_days,
    wind_direction,
)

"""Tests that correspond to wind.py"""

def test_wind_direction(dataarray_u, dataarray_v):
    u = dataarray_u.sel({"variable": "u"}, drop=True)
    v = dataarray_v.sel({"variable": "v"}, drop=True)
    u = u.transpose("time", "lat", "lon")
    v = v.transpose("time", "lat", "lon")
    assert type(wind_direction(u, v)) == xr.DataArray


def test_wind_speed_anomalies(dataarray_u: xr.DataArray):
    ws = dataarray_u
    clim = dataarray_u

    ws = ws.sel({"variable": "u"}, drop=True)
    clim = clim.sel({"variable": "u"}, drop=True)

    ws = ws.transpose("time", "lat", "lon")
    clim = clim.transpose("time", "lat", "lon")

    # assert anomalies = 0 if compared to the same dataset
    assert wind_speed_anomalies(ws, clim).all() == 0


def test_wind_power_density(dataarray_u: xr.DataArray):
    u = dataarray_u.sel({"variable": "u"}, drop=True)
    u = u.transpose("time", "lat", "lon")
    assert wind_power_density(u).all()


def test_wind_speed_at_height(dataarray_u: xr.DataArray, dataarray_v: xr.DataArray):
    u = dataarray_u.sel({"variable": "u"}, drop=True)
    v = dataarray_v.sel({"variable": "v"}, drop=True)
    u = u.transpose("time", "lat", "lon")
    v = v.transpose("time", "lat", "lon")
    assert wind_speed_at_height(u, v, 100.0).all()


def test_power_output(dataarray_u: xr.DataArray):
    ws = dataarray_u.sel(
        {"variable": "u"}, drop=True
    )  # trick to pass some data # todo: pass actual ws
    a = power_output(
        ws=ws.values.flatten(),
        iec_class="I",
        rated_power=2.0,
        cut_in_speed=1.0,
        rated_speed=4.0,
        cut_out_speed=6.0,
    ).all()
    # v = dataarray_v.sel({'variable':'v'}, drop=True)
    assert type(a) == xr.DataArray
    assert all(a.values.flatten()) <= 1.0
    assert all(a.values.flatten()) >= 0


def test_capacity_factor(dataarray_u: xr.DataArray):
    u = dataarray_u.sel({"variable": "u"}, drop=True)
    u = u.transpose("time", "lat", "lon")
    cfI = capacity_factor(u, iec_class="I")
    cfII = capacity_factor(u, iec_class="II")
    cfIII = capacity_factor(u, iec_class="III")
    cfS = capacity_factor(u, iec_class="S")
    assert ((cfI >= 0) & (cfI <= 1)).all(), "Some elements are out of the range [0, 1]"
    assert (
        (cfII >= 0) & (cfII <= 1)
    ).all(), "Some elements are out of the range [0, 1]"
    assert (
        (cfIII >= 0) & (cfIII <= 1)
    ).all(), "Some elements are out of the range [0, 1]"
    assert ((cfS >= 0) & (cfS <= 1)).all(), "Some elements are out of the range [0, 1]"


def test_capacity_factor_histogram(dataarray_u):
    pass

def test_capacity_factor_histogram_1d(dataarray_u: xr.DataArray):
    u = dataarray_u.sel({"variable": "u"}, drop=True)
    u = u.transpose("time", "lat", "lon")
    assert capacity_factor_histogram_1d(
        u, bins=5, target_lat=1.0, target_lon=1.0, iec_class="I"
    )


def test_wind_speed_histogram_1d(dataarray_u: xr.DataArray):
    u = dataarray_u.sel({"variable": "u"}, drop=True)
    u = u.transpose("time", "lat", "lon")
    assert wind_speed_histogram_1d(u, bins=5, target_lat=1.0, target_lon=1.0)


def test_annual_energy_production_wind(dataarray_u: xr.DataArray):
    # Create a time dimension with 24 hourly intervals
    times = pd.date_range("2023-01-01", periods=24, freq="h")

    # Generate 24 positive random values (e.g., between 1 and 10)
    data = np.random.rand(24) * 9 + 1  # This scales the values to be between 1 and 10

    # Create the DataArray
    cf = xr.DataArray(data, coords=[times], dims=["time"])

    epw = annual_energy_production_wind(cf, rated_power=2.0, num_turbines=1)
    epw2 = annual_energy_production_wind(cf, rated_power=2.0, num_turbines=2)
    assert epw.all()
    assert (epw * 2).all() == epw2.all()


def test_high_wind_events(dataarray_u: xr.DataArray):
    u = dataarray_u.sel({"variable": "u"}, drop=True)
    total = high_wind_events(u, 10)['hwe'].sum()
    assert total >= 0
    assert high_wind_events(u, 10, None)


def test_low_wind_events(dataarray_u: xr.DataArray):
    u = dataarray_u.sel({"variable": "u"}, drop=True)
    total = low_wind_events(u, 10)['lwe'].sum()
    assert total >= 0
    assert low_wind_events(u, 10, None)


def test_calm_days(dataarray_u: xr.DataArray):
    u = dataarray_u.sel({"variable": "u"}, drop=True)
    total = calm_days(u, 10)['cd'].sum()
    assert total >= 0
    assert calm_days(u, 10, None)


def test_windy_days(dataarray_u: xr.DataArray):
    u = dataarray_u.sel({"variable": "u"}, drop=True)
    total = windy_days(u, 10)['wd'].sum()
    assert total >= 0
    assert windy_days(u, 10, None)


def test_capacity_factor_histogram_opa():
    cf_i = xr.open_dataset('test_data/1990_01_01_T00_00_cf_I.nc')  # idataarray_u.sel({"variable": "u"}, drop=True)
    capacity_factor_histogram_opa(cf_i, mask="None", working_dir="test_data/")
