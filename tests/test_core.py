import numpy as np
import pytest
import xarray as xr

from energy_onshore.core import (
    check_temperature,
    convert_temperature,
    wind_speed,
    cosine_sza_hourly,
    percentile,
    moving_average,
    temporal_rescaling,
    select_region,
    temporal_rescaling,
    select_point,
    create_dataset,
    get_type,
    check_dims,
)

"""Tests that correspond to core.py @froura"""


def test_check_temperature(dataarray_t_k, dataarray_t_c):
    assert check_temperature(dataarray_t_k) == "K"
    assert check_temperature(dataarray_t_c) == "C"


def test_convert_temperature(dataarray_t_k, dataarray_t_c):
    assert (
        convert_temperature(dataarray_t_k, unit="C").all() == dataarray_t_c.values.all()
    )
    assert (
        convert_temperature(dataarray_t_c, unit="K").all() == dataarray_t_k.values.all()
    )


def test_wind_speed(dataarray_u, dataarray_v):
    u = dataarray_u.sel({"variable": "u"}, drop=True)
    v = dataarray_v.sel({"variable": "v"}, drop=True)
    u = u.transpose("time", "lat", "lon")
    v = v.transpose("time", "lat", "lon")
    # TODO: assert to some precomputed dataarray
    assert type(wind_speed(u, v)) == xr.DataArray


def test_cosine_sza_hourly(dataarray_u, dataarray_v):
    start_date = np.datetime64(1, "D")
    end_date = np.datetime64(10, "D")
    lats = dataarray_u["lat"].data
    lons = dataarray_u["lon"].data
    assert (
        cosine_sza_hourly(start_date, end_date, lats, lons).all()
        == xr.DataArray().all()
    )


def test_percentile(dataarray_t_c):
    assert percentile(dataarray_t_c, wanted_percentile=float(50), axis=0).all()


def test_moving_average(
    dataarray_t_c, val=1
):  # TODO: add a warning when val=> than the lenght of the data
    assert moving_average(dataarray_t_c.data, 1).all()


def test_temporal_rescaling(dataarray_t_c, scale="daily"):
    assert temporal_rescaling(dataarray_t_c, scale="daily").all()


def test_select_region(dataarray_t_c, l0l0=[0, 0], l1l1=[1, 1]):
    assert select_region(dataarray_t_c, l0l0, l1l1).all()
    region = select_region(dataarray_t_c, l0l0, l1l1)
    assert type(region) == xr.DataArray


def test_select_point(dataarray_t_c, lat=float(0), lon=float(0)):
    assert select_point(dataarray_t_c, lat, lon).all()
    point = select_point(dataarray_t_c, lat, lon)
    assert type(point) == xr.DataArray


def test_create_dataset():
    # TODO: check if we are running in circles. This is used in test_core.py
    pass


def test_get_type(dataarray_t_c, val1=int(1), val2=float(1)):
    assert get_type(val1) == "int"
    assert get_type(val2) == "float"
    assert get_type(dataarray_t_c) == "DataArray"


def test_check_dims(dataarray_t_c):
    with pytest.raises(ValueError):
        check_dims(dataarray_t_c)
    t_c = dataarray_t_c.sel({"variable": "t_c"}, drop=True)
    assert check_dims(t_c) == None

