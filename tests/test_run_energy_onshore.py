#!/usr/bin/env python3
"""
Definition of the runs cripts to be run in the workflow runscript
"""

# Load libraries
import os
from datetime import datetime, timezone
import importlib

import numpy as np
import xarray as xr
import pandas as pd

from energy_onshore.run_energy_onshore import (
    get_time_utc,
    get_application_version,
    run_wind_speed_anomalies,
    run_capacity_factor_i,
    run_capacity_factor_ii,
    run_capacity_factor_iii,
    run_capacity_factor_s,
    run_cdd,
    run_hdd,
    run_high_wind_events,
    run_low_wind_events,
    run_pv_potential,
    run_capacity_factor_histogram_opa,
    run_wind_direction,
)

"""Tests that correspond to run_energy_onshore.py @froura"""

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
maskfile = "test_data/testmask.nc"

# test get time UTC
def test_get_time_utc():
    assert get_time_utc()

# test get app version
def test_get_application_version():
    assert get_application_version()

# Wind direction
def test_run_wind_direction(
    iniyear=iniyear,
    inimonth=inimonth,
    iniday=iniday,
    in_path=in_path,
    out_path=out_path,
    hpcprojdir=hpcprojdir,
    maskfile=maskfile,
):
    assert (
        run_wind_direction(
            iniyear, inimonth, iniday, in_path, out_path, mask=maskfile
        )
        is None
    )
    assert (
        run_wind_direction(
            iniyear, inimonth, iniday, in_path, out_path, mask=None
        )
        is None
    )


# Wind speed anomalies
# def test_run_wind_speed_anomalies(iniyear=iniyear, inimonth=inimonth, iniday=iniday, in_path=in_path, out_path=out_path, hpcprojdir=hpcprojdir):
#    assert run_wind_speed_anomalies(iniyear, inimonth, iniday, in_path, out_path, hpcprojdir)

# assert that the functions are executed without obvious errors


# Capacity factor (class I)
def test_run_capacity_factor_i(
    iniyear=iniyear,
    inimonth=inimonth,
    iniday=iniday,
    finyear=finyear,
    finmonth=finmonth,
    finday=finday,
    in_path=in_path,
    out_path=out_path,
    maskfile=maskfile,
):
    assert (
        run_capacity_factor_i(
            iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path, mask=maskfile
        )
        is None
    )
    assert (
        run_capacity_factor_i(
            iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path, mask=None
        )
        is None
    )

# Capacity factor (class II)
def test_run_capacity_factor_ii(
    iniyear=iniyear,
    inimonth=inimonth,
    iniday=iniday,
    finyear=finyear,
    finmonth=finmonth,
    finday=finday,
    in_path=in_path,
    out_path=out_path,
    maskfile=maskfile,
):
    assert (
        run_capacity_factor_ii(
            iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path, mask=maskfile
        )
        is None
    )
    assert (
        run_capacity_factor_ii(
            iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path, mask=None
        )
        is None
    )


# Capacity factor (class III)
def test_run_capacity_factor_iii(
    iniyear=iniyear,
    inimonth=inimonth,
    iniday=iniday,
    finyear=finyear,
    finmonth=finmonth,
    finday=finday,
    in_path=in_path,
    out_path=out_path,
    maskfile=maskfile,
):
    assert (
        run_capacity_factor_iii(
            iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path, mask=maskfile
        )
        is None
    )
    assert (
        run_capacity_factor_iii(
            iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path, mask=None
        )
        is None
    )


# Capacity factor (class S)
def test_run_capacity_factor_s(
    iniyear=iniyear,
    inimonth=inimonth,
    iniday=iniday,
    finyear=finyear,
    finmonth=finmonth,
    finday=finday,
    in_path=in_path,
    out_path=out_path,
    maskfile=maskfile,
):
    assert (
        run_capacity_factor_s(
            iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path, mask=maskfile
        )
        is None
    )
    assert (
        run_capacity_factor_s(
            iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path, mask=None
        )
        is None
    )


# Cooling degree days (CDD)
def test_run_cdd(
    iniyear=iniyear,
    inimonth=inimonth,
    iniday=iniday,
    in_path=in_path,
    out_path=out_path,
    maskfile=maskfile,
):
    assert run_cdd(iniyear, inimonth, iniday, in_path, out_path, mask=maskfile) is None
    assert run_cdd(iniyear, inimonth, iniday, in_path, out_path, mask=None) is None


# Heating degree days (HDD)
def test_run_hdd(
    iniyear=iniyear,
    inimonth=inimonth,
    iniday=iniday,
    in_path=in_path,
    out_path=out_path,
    maskfile=maskfile,
):
    assert run_hdd(iniyear, inimonth, iniday, in_path, out_path, mask=maskfile) is None
    assert run_hdd(iniyear, inimonth, iniday, in_path, out_path, mask=None) is None


# High wind events
def test_run_high_wind_events(
    iniyear=iniyear,
    inimonth=inimonth,
    iniday=iniday,
    finyear=finyear,
    finmonth=finmonth,
    finday=finday,
    in_path=in_path,
    out_path=out_path,
    maskfile=maskfile,
):
    assert (
        run_high_wind_events(
            iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path, mask=maskfile
        )
        is None
    )
    assert (
        run_high_wind_events(
            iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path, mask=None
        )
        is None
    )


# Low wind events
def test_run_low_wind_events(
    iniyear=iniyear,
    inimonth=inimonth,
    iniday=iniday,
    finyear=finyear,
    finmonth=finmonth,
    finday=finday,
    in_path=in_path,
    out_path=out_path,
    maskfile=maskfile,
):
    assert (
        run_low_wind_events(
            iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path, mask=maskfile
        )
        is None
    )
    assert (
        run_low_wind_events(
            iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path, mask=None
        )
        is None
    )
    
def test_run_capacity_factor_histogram_opa(
    iniyear=iniyear,
    inimonth=inimonth,
    iniday=iniday,
    finyear=finyear,
    finmonth=finmonth,
    finday=finday,
    in_path=in_path,
    out_path=out_path,
):
    run_capacity_factor_histogram_opa(
        iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path, cf_type='I'
    )
    run_capacity_factor_histogram_opa(
        iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path, cf_type='II'
    )
    assert (
        run_capacity_factor_histogram_opa(
            iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path
        )
        is None
    )

# pv_potential
def test_run_pv_potential(
    iniyear=iniyear,
    inimonth=inimonth,
    iniday=iniday,
    finyear=finyear,
    finmonth=finmonth,
    finday=finday,
    in_path=in_path,
    out_path=out_path,
    maskfile=maskfile,
):
    run_low_wind_events(
            iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path, mask=maskfile
        )
    assert (
        run_low_wind_events(
            iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path, mask=None
        )
        is None
    )
