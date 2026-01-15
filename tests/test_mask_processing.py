import xarray as xr
import numpy as np
import pandas as pd

from energy_onshore.mask_processing import (
    onshore_mask,
)

"""Tests that correspond to land-sea mask.py"""

# define arguments:
iniyear = "1990"
inimonth = "01"
iniday = "01"
in_path = "test_data/"
out_path = "."
hpcprojdir = "test_data/"
finyear = "1990"
finmonth = "01"
finday = "01"

u100_file = (
        f"{in_path}/{iniyear}_{inimonth}_{iniday}_T00_00_to_{finyear}_{finmonth}_{finday}_"
        "T23_00_u_raw_data.nc"
        )

variable_data=xr.open_dataset(u100_file).u[0]
mask_file=f"{in_path}/testmask.nc" #testtest_mask.nc" #testmask.nc"

# test that a mask with all ocean values returns all nans.
def test_onshore_mask(): 
    output=onshore_mask(variable_data, mask_file)
    output.to_netcdf('tmp.nc')
    assert output.shape == variable_data.shape
