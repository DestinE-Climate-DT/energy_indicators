import xarray as xr
import numpy as np
import pandas as pd

from energy_onshore import (
    cooling_degree_days,
    heating_degree_days,
)
from energy_onshore.mask_processing import (
    onshore_mask,
)

"""Tests that correspond to demand.py"""

def test_cooling_degree_days(dataarray_t_c: xr.DataArray):
    t_c = dataarray_t_c.sel({"variable": "t_c"}, drop=True)
    tm = t_c.copy()
    tx = t_c + 5
    tn = t_c - 5
    total = cooling_degree_days(tm, tx, tn, base=22.0, mask=None)
    assert total['cdd'].all() >= 0

def test_heating_degree_days(dataarray_t_c: xr.DataArray):
    t_c = dataarray_t_c.sel({"variable": "t_c"}, drop=True)
    tm = t_c.copy()
    tx = t_c + 5
    tn = t_c - 5
    total = heating_degree_days(tm, tx, tn, base=15.5, mask=None)
    assert total['hdd'].all() >= 0
