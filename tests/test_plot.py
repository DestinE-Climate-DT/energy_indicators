"""
Tests for the ploting functions
"""

import xarray as xr
import numpy as np
import pandas as pd

from energy_onshore.plot import define_colormaps, load_plot_libs, plot_cf, plot_degree_days, plot_lwe_hwe, plot_map


def test_load_plot_libs():
    assert load_plot_libs() is None
    
# Define colormaps
def test_define_colormaps():
    pass
    #assert define_colormaps
    
# Updated plot_map function
def test_plot_map():
    pass
    #assert plot_map

# Define the plot_cf function
def test_plot_cf():
    pass
    #assert plot_cf

# Define the plot_lwe_hwe function
def test_plot_lwe_hwe():
    pass
    #assert plot_lwe_hwe

# define plotting for cdd and hdd days
def test_plot_degree_days():
    pass
    #assert plot_degree_days
