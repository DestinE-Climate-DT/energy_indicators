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

from energy_indicators.run_energy_indicators import (
    get_time_utc,
    get_application_version,
    run_wind_speed_anomalies,
    run_capacity_factor_i,
    run_capacity_factor_ii,
    run_capacity_factor_iii,
    run_capacity_factor_s,
    run_cdd,
    run_hdd,
    run_wind_producing_regimes,
    run_pv_potential,
    run_capacity_factor_histogram_opa,
    run_wind_direction,
    run_wind_power_density,
    run_daily_cell_temp,
)

"""Tests that correspond to run_energy_indicators.py @froura"""

# define arguments:
iniyear = "1990"
inimonth = "01"
iniday = "01"
in_path = "test_data/"
out_path = "test_data/"
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
    tmp_path,
    iniyear=iniyear,
    inimonth=inimonth,
    iniday=iniday,
    in_path=in_path,
    out_path=out_path,
    hpcprojdir=hpcprojdir,
    maskfile=maskfile,
):
    output_dir = tmp_path / "test_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    assert (
        run_wind_direction(iniyear, inimonth, iniday, in_path, output_dir, mask=maskfile)
        is None
    )
    assert (
        run_wind_direction(iniyear, inimonth, iniday, in_path, output_dir, mask=None)
        is None
    )


# Wind speed anomalies
# def test_run_wind_speed_anomalies(iniyear=iniyear, inimonth=inimonth, iniday=iniday, in_path=in_path, out_path=out_path, hpcprojdir=hpcprojdir):
#    assert run_wind_speed_anomalies(iniyear, inimonth, iniday, in_path, out_path, hpcprojdir)

# assert that the functions are executed without obvious errors


# Capacity factor (class I)
def test_run_capacity_factor_i(
    tmp_path,
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
    output_dir = tmp_path / "test_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    assert (
        run_capacity_factor_i(
            iniyear,
            inimonth,
            iniday,
            finyear,
            finmonth,
            finday,
            in_path,
            output_dir,
            mask=maskfile,
        )
        is None
    )
    assert (
        run_capacity_factor_i(
            iniyear,
            inimonth,
            iniday,
            finyear,
            finmonth,
            finday,
            in_path,
            output_dir,
            mask=None,
        )
        is None
    )


# Capacity factor (class II)
def test_run_capacity_factor_ii(
    tmp_path,
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
    output_dir = tmp_path / "test_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    assert (
        run_capacity_factor_ii(
            iniyear,
            inimonth,
            iniday,
            finyear,
            finmonth,
            finday,
            in_path,
            output_dir,
            mask=maskfile,
        )
        is None
    )
    assert (
        run_capacity_factor_ii(
            iniyear,
            inimonth,
            iniday,
            finyear,
            finmonth,
            finday,
            in_path,
            output_dir,
            mask=None,
        )
        is None
    )


# Capacity factor (class III)
def test_run_capacity_factor_iii(
    tmp_path,
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
    output_dir = tmp_path / "test_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    assert (
        run_capacity_factor_iii(
            iniyear,
            inimonth,
            iniday,
            finyear,
            finmonth,
            finday,
            in_path,
            output_dir,
            mask=maskfile,
        )
        is None
    )
    assert (
        run_capacity_factor_iii(
            iniyear,
            inimonth,
            iniday,
            finyear,
            finmonth,
            finday,
            in_path,
            output_dir,
            mask=None,
        )
        is None
    )


# Capacity factor (class S)
def test_run_capacity_factor_s(
    tmp_path,
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
    output_dir = tmp_path / "test_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    assert (
        run_capacity_factor_s(
            iniyear,
            inimonth,
            iniday,
            finyear,
            finmonth,
            finday,
            in_path,
            output_dir,
            mask=maskfile,
        )
        is None
    )
    assert (
        run_capacity_factor_s(
            iniyear,
            inimonth,
            iniday,
            finyear,
            finmonth,
            finday,
            in_path,
            output_dir,
            mask=None,
        )
        is None
    )

# Heating degree days (HDD)
def test_run_hdd(
    tmp_path,
    iniyear=iniyear,
    inimonth=inimonth,
    iniday=iniday,
    in_path=in_path,
    out_path=out_path,
    maskfile=maskfile,
):
    output_dir = tmp_path / "test_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    assert run_hdd(iniyear, inimonth, iniday, in_path, output_dir, mask=maskfile) is None
    assert run_hdd(iniyear, inimonth, iniday, in_path, output_dir, mask=None) is None

# Cooling degree days (CDD)
def test_run_cdd(
    tmp_path,
    iniyear=iniyear,
    inimonth=inimonth,
    iniday=iniday,
    in_path=in_path,
    out_path=out_path,
    maskfile=maskfile,
):
    output_dir = tmp_path / "test_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    assert run_cdd(iniyear, inimonth, iniday, in_path, output_dir, mask=maskfile) is None
    assert run_cdd(iniyear, inimonth, iniday, in_path, output_dir, mask=None) is None

# Wind producing regimes
def test_run_wind_producing_regimes(
    tmp_path,
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
    output_dir = tmp_path / "test_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    assert (
        run_wind_producing_regimes(
            iniyear,
            inimonth,
            iniday,
            finyear,
            finmonth,
            finday,
            in_path,
            output_dir,
            mask=maskfile,
        )
        is None
    )
    assert (
        run_wind_producing_regimes(
            iniyear,
            inimonth,
            iniday,
            finyear,
            finmonth,
            finday,
            in_path,
            output_dir,
            mask=None,
        )
        is None
    )

def test_run_capacity_factor_histogram_opa(
    tmp_path,
    iniyear=iniyear,
    inimonth=inimonth,
    iniday=iniday,
    in_path=in_path,
    out_path=out_path,
):
    output_dir = tmp_path / "test_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_capacity_factor_histogram_opa(
        iniyear,
        inimonth,
        iniday,
        in_path,
        output_dir,
        cf_type="I",
        nworkers=2,
    )
    run_capacity_factor_histogram_opa(
        iniyear,
        inimonth,
        iniday,
        in_path,
        output_dir,
        cf_type="II",
        nworkers=1,
    )
    assert (
        run_capacity_factor_histogram_opa(
            iniyear, inimonth, iniday, in_path, output_dir
        )
        is None
    )


# pv_potential
def test_run_pv_potential(
    tmp_path,
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
    output_dir = tmp_path / "test_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_pv_potential(
        iniyear,
        inimonth,
        iniday,
        finyear,
        finmonth,
        finday,
        in_path,
        output_dir,
        mask=maskfile,
    )
    assert (
        run_pv_potential(
            iniyear,
            inimonth,
            iniday,
            finyear,
            finmonth,
            finday,
            in_path,
            output_dir,
            mask=None,
        )
        is None
    )

# Wind power density
def test_run_wind_power_density(
    tmp_path,
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
    output_dir = tmp_path / "test_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_wind_power_density(
        iniyear,
        inimonth,
        iniday,
        finyear,
        finmonth,
        finday,
        in_path,
        output_dir,
        mask=maskfile,
    )

    assert (
        run_wind_power_density(
            iniyear,
            inimonth,
            iniday,
            finyear,
            finmonth,
            finday,
            in_path,
            output_dir,
            mask=None,
        )
        is None
    )

# NUTS spatial aggregation
def test_run_nuts_aggregation(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box
    from energy_indicators.aggregation import build_region_mask, build_lat_weights
    from energy_indicators.run_energy_indicators import run_nuts_aggregation

    # Tiny synthetic region set, two countries with a known border
    lat = np.linspace(40, 45, 10)
    lon = np.linspace(0, 5, 10)
    shp_path = str(tmp_path / "test_regions.shp")
    gdf = gpd.GeoDataFrame(
        {"NUTS_ID": ["AA", "BB"]},
        geometry=[box(0, 40, 2.5, 45), box(2.5, 40, 5, 45)],
        crs="EPSG:4326",
    )
    gdf.to_file(shp_path)

    region_mask = build_region_mask(lat, lon, shp_path, id_col="NUTS_ID")
    region_mask_file = str(tmp_path / "region_mask.nc")
    region_mask.to_dataset(name="mask").to_netcdf(region_mask_file)

    lat_weights_file = str(tmp_path / "lat_weights.nc")
    build_lat_weights(lat, lon).to_dataset(name="lat_weights").to_netcdf(lat_weights_file)

    # An already-computed indicator file, matching the naming/shape
    # convention run_wind_power_density (and the other run_* functions)
    # already write, since run_nuts_aggregation reads that output rather
    # than raw met variables
    time = xr.date_range("1990-01-01", periods=1)
    wpd = xr.DataArray(
        np.random.uniform(50, 400, (1, len(lat), len(lon))),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name="wpd",
    )
    wpd.to_dataset().to_netcdf(str(tmp_path / "1990_01_01_T00_00_wpd.nc"))

    assert (
        run_nuts_aggregation(
            "1990", "01", "01", "wpd",
            str(tmp_path), str(tmp_path),
            region_mask_file, lat_weights_file,
            stat="mean", region_level=0,
        )
        is None
    )

    output_file = tmp_path / "1990_01_01_T00_00_wpd_nuts0.nc"
    assert output_file.exists()

    result = xr.open_dataset(str(output_file))
    assert "region" in result.dims
    assert result.sizes["region"] == 2
    assert set(result.region.values) == {"AA", "BB"}
    assert bool(result["wpd"].notnull().all())
    
#daily cell temperature
def test_run_daily_cell_temp(
    tmp_path,
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
    output_dir = tmp_path / "test_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_daily_cell_temp(
        iniyear,
        inimonth,
        iniday,
        finyear,
        finmonth,
        finday,
        in_path,
        output_dir,
        mask=maskfile,
    )
    assert (
        run_daily_cell_temp(
            iniyear,
            inimonth,
            iniday,
            finyear,
            finmonth,
            finday,
            in_path,
            output_dir,
            mask=None,
        )
        is None
    )
