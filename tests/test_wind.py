import xarray as xr
import numpy as np
import pandas as pd

from energy_onshore.wind import (
    wind_speed_anomalies,
    wind_power_density,
    wind_speed_at_height,
    power_curve,
    capacity_factor,
    capacity_factor_histogram_1d,
    wind_speed_histogram_1d,
    annual_energy_production_wind,
    high_wind_events,
    low_wind_events,
    cooling_degree_days,
    heating_degree_days,
)

"""Tests that correspond to wind.py @froura"""


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


def test_power_curve():
    a = power_curve(
        iec_class="I",
        rated_power=2.0,
        cut_in_speed=1.0,
        rated_speed=4.0,
        cut_out_speed=6.0,
    ).all()
    # v = dataarray_v.sel({'variable':'v'}, drop=True)
    assert type(a) == xr.DataArray  # assert result==xr.DataArray


def test_capacity_factor(dataarray_u: xr.DataArray):
    u = dataarray_u.sel({"variable": "u"}, drop=True)
    u = u.transpose("time", "lat", "lon")
    cfI = capacity_factor(u, iec_class="I")
    cfII = capacity_factor(u, iec_class="II")
    cfIII = capacity_factor(u, iec_class="III")
    cfS = capacity_factor(u, iec_class="S")
    assert ((cfI >= 0) & (cfI <= 1)).all(), "Some elements are out of the range [0, 1]"
    assert ((cfII >= 0) & (cfII <= 1)).all(), "Some elements are out of the range [0, 1]"
    assert ((cfIII >= 0) & (cfIII <= 1)).all(), "Some elements are out of the range [0, 1]"
    assert ((cfS >= 0) & (cfS <= 1)).all(), "Some elements are out of the range [0, 1]"

#def test_capacity_factor_histogram(dataarray_u):
#    u = dataarray_u.sel({"variable": "u"}, drop=True)
#    u = u.transpose("time", "lat", "lon")    #u=dataarray_u
#    capacity_factor_histogram(u, 5,iec_class="I")
#    assert u 



def test_capacity_factor_histogram_1d(dataarray_u: xr.DataArray):
    u = dataarray_u.sel({"variable": "u"}, drop=True)
    u = u.transpose("time", "lat", "lon")
    assert capacity_factor_histogram_1d(u, bins=5, target_lat=1., target_lon=1., iec_class="I")

    
#def test_wind_speed_histogram(dataarray_u):
#    u = dataarray_u.sel({"variable": "u"}, drop=True)
#    u = u.transpose("time", "lat", "lon")    #u=dataarray_u
#    assert wind_speed_histogram(u, bins=5)


def test_wind_speed_histogram_1d(dataarray_u: xr.DataArray):
    u = dataarray_u.sel({"variable" : "u"}, drop=True)
    u = u.transpose("time", "lat", "lon")
    assert wind_speed_histogram_1d(u, bins=5, target_lat=1., target_lon=1.)

def test_annual_energy_production_wind(dataarray_u: xr.DataArray):
    # Create a time dimension with 24 hourly intervals
    times = pd.date_range('2023-01-01', periods=24, freq='h')

    # Generate 24 positive random values (e.g., between 1 and 10)
    data = np.random.rand(24) * 9 + 1  # This scales the values to be between 1 and 10

    # Create the DataArray
    cf = xr.DataArray(data, coords=[times], dims=["time"])
    
    epw=annual_energy_production_wind(cf, rated_power=2., num_turbines=1)
    epw2=annual_energy_production_wind(cf, rated_power=2., num_turbines=2)
    assert epw.all()    
    assert (epw*2).all() == epw2.all() 


def test_high_wind_events(dataarray_u: xr.DataArray):
    u = dataarray_u.sel({"variable": "u"}, drop=True)
    total = high_wind_events(u, 10).data.sum()
    assert total >= 0


def test_low_wind_events(dataarray_u: xr.DataArray):
    u = dataarray_u.sel({"variable": "u"}, drop=True)
    total = low_wind_events(u, 10).data.sum()
    assert total >= 0

def test_cooling_degree_days(dataarray_t_c: xr.DataArray):
    t_c = dataarray_t_c.sel({"variable": "t_c"}, drop=True)
    tm=t_c.copy()
    tx=t_c+5
    tn=t_c-5
    total =cooling_degree_days(tm, tx, tn, base=22.0) 
    assert len(total) ==2
    assert total[0].data.all() >= 0
    

def test_heating_degree_days(dataarray_t_c: xr.DataArray):
    t_c = dataarray_t_c.sel({"variable": "t_c"}, drop=True)
    tm=t_c.copy()
    tx=t_c+5
    tn=t_c-5
    total =heating_degree_days(tm, tx, tn, base=15.5) 
    assert len(total) ==2
    assert total[0].data.all() >= 0
