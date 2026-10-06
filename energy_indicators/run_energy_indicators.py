#!/usr/bin/env python3
"""
| Destination Earth: Energy Indicators application

Wrapper functions that tie the lower-level indicator functions together:
reading raw input NetCDF files, calling the relevant computation function,
attaching global metadata attributes, and writing output NetCDF files.
This is the layer the ClimateDT workflow calls directly.
"""

# External libraries
import importlib
import os
from datetime import datetime, timezone

import numpy as np
import xarray as xr
import pandas as pd

# Internal libraries
from energy_indicators.core import wind_speed, convert_temperature

from energy_indicators.demand import (
    heating_degree_days,
    cooling_degree_days,
)

from energy_indicators.wind import (
    wind_speed_anomalies,
    capacity_factor,
    high_wind_events,
    low_wind_events,
    capacity_factor_histogram_opa,
    wind_direction,
    wind_power_density,
    daily_wind_power_density,
)

from energy_indicators.solar import pv_pot, cell_temp, daily_cell_temp, daily_pv_potential

from energy_indicators.aggregation import (
    load_region_mask,
    load_lat_weights,
    build_lat_weights,
    aggregate_region,
)


# get time UTC


def get_time_utc():  # add pytest
    """
    Parameters
    ----------
    -

    Returns
    -------
    formatted_time : str
        Current UTC date and time, formatted as "YYYY-MM-DD HH:MM:SS".
    """
    # Get the current time in UTC
    current_time_utc = datetime.now(timezone.utc)

    # Format the time as a string in the specified format
    formatted_time = current_time_utc.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time


def get_application_version():
    """
    Parameters
    ----------
    -

    Returns
    -------
    version : str
        The installed energy_indicators package version.
    """
    version = importlib.metadata.version("energy_indicators")

    return version


def _packed_int16_encoding(ds, scale_factor=0.01, add_offset=0.0):
    """
    Build NetCDF encoding to store float variables as packed int16.

    Uses a fixed ``scale_factor`` and ``add_offset`` (default 0.0). With a
    fixed scale, a zero offset works better for non-negative indicators
    (CF, PVP, CDD, HDD) than a mid-range offset, which can unpack small
    values slightly below zero. On read, values unpack as::

        value = packed * scale_factor + add_offset

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Dataset or DataArray whose data variable(s) will be packed.
    scale_factor : float, optional
        Packing resolution (default 0.01).
    add_offset : float, optional
        Packing offset (default 0.0).

    Returns
    -------
    encoding : dict
        Encoding dict suitable for ``Dataset.to_netcdf(..., encoding=...)``.
    """
    var_encoding = {
        "dtype": "int16",
        "scale_factor": np.float64(scale_factor),
        "add_offset": np.float64(add_offset),
        "_FillValue": np.int16(-32767),
    }
    if isinstance(ds, xr.DataArray):
        name = ds.name if ds.name is not None else "data"
        return {name: var_encoding}
    return {name: var_encoding for name in ds.data_vars}


# Wind direction


def run_wind_direction(iniyear, inimonth, iniday, in_path, out_path, mask=None):
    """
    Parameters
    ----------
    iniyear : string
        initial year of the streamed data YYYY.
    inimonth : string
        initial month of the streamed data MM.
    iniday : string
        initial day of the streamed data DD.
    in_path : string
        root path where to get the data from.
    out_path : string
        path where the output data goes to.
    mask : str or None (None by default).
        Path to a netCDF file containing a land-sea mask. If provided, the mask will be applied.

    Returns
    -------
    None.
    """

    # Provide the data file name for all variables
    u100_file = (
        f"{iniyear}_{inimonth}_{iniday}_T00_00_to_{iniyear}_{inimonth}_{iniday}_"
        "T23_00_u_raw_data.nc"
    )
    v100_file = (
        f"{iniyear}_{inimonth}_{iniday}_T00_00_to_{iniyear}_{inimonth}_{iniday}_"
        "T23_00_v_raw_data.nc"
    )

    absolute_path_u100 = os.path.join(in_path, u100_file)
    absolute_path_v100 = os.path.join(in_path, v100_file)

    data_u100 = xr.open_dataset(absolute_path_u100)
    data_v100 = xr.open_dataset(absolute_path_v100)

    # time
    time = get_time_utc()

    # version
    version = get_application_version()

    message = (
        time + " ENERGY: wind direction computed using the "
        f" energy_indicators application v{version}."
    )

    history = data_u100.attrs["history"] + data_v100.attrs["history"] + message

    # Import processing script.

    u100 = data_u100["u"][:, 0, :, :]
    v100 = data_v100["v"][:, 0, :, :]

    wind_dir = wind_direction(u100, v100, mask=mask)

    # Global attrs:
    wind_dir.attrs = {
        "resolution": data_u100.attrs["resolution"],
        "generation": data_u100.attrs["generation"],
        "activity": data_u100.attrs["activity"],
        "dataset": data_u100.attrs["dataset"],
        "stream": data_u100.attrs["stream"],
        "model": data_u100.attrs["model"],
        "experiment": data_u100.attrs["experiment"],
        "levtype": data_u100.attrs["levtype"],
        "expver": data_u100.attrs["expver"],
        "class": data_u100.attrs["class"],
        "type": data_u100.attrs["type"],
        "realization": data_u100.attrs["realization"],
    }

    wind_dir.attrs["history"] = history

    date = pd.to_datetime(wind_dir["time"].values[0])
    yyyy_mm_dd = date.strftime("%Y_%m_%d")

    output_file_path = os.path.join(out_path, f"{yyyy_mm_dd}_T00_00_wd.nc")

    # reduce file size by converting to float32
    wind_dir = wind_dir.astype(np.float32)

    wind_dir.to_netcdf(path=output_file_path, mode="w", encoding=_packed_int16_encoding(wind_dir, scale_factor=0.1))
    print("Wind direction has been produced and saved to: ", output_file_path)


# Wind speed anomalies


def run_wind_speed_anomalies(
    iniyear, inimonth, iniday, in_path, out_path, hpcprojdir, mask=None
):
    """
    Parameters
    ----------
    iniyear : string
        initial year of the streamed data YYYY.
    inimonth : string
        initial month of the streamed data MM.
    iniday : string
        initial day of the streamed data DD.
    in_path : string
        root path where to get the data from.
    out_path : string
        path where the output data goes to.
    hpcprojdir : string
        project path in the HPC.
    mask : str or None (None by default).
        Path to a netCDF file containing a land-sea mask. Note: this
        parameter is currently not forwarded to the underlying
        computation, masking has no effect.

    Returns
    -------
    None.
    """

    # Provide the data file name for all variables

    # adapt this for 1 and several days run:
    u100_file = f"{iniyear}_{inimonth}_{iniday}_u_timestep_60_daily_mean.nc"
    v100_file = f"{iniyear}_{inimonth}_{iniday}_v_timestep_60_daily_mean.nc"

    absolute_path_u100 = os.path.join(in_path, u100_file)
    absolute_path_v100 = os.path.join(in_path, v100_file)

    data_u100 = xr.open_dataset(absolute_path_u100)
    data_v100 = xr.open_dataset(absolute_path_v100)

    # time
    time = get_time_utc()

    # version
    version = get_application_version()

    message = (
        time + " ENERGY: wind speed anomalies computed using the "
        f"energy_indicators application v{version}."
    )

    history = data_u100.attrs["history"] + data_v100.attrs["history"] + message

    # Import processing script.

    u100 = data_u100["u"][:, 0, 1:, :-1]
    v100 = data_v100["v"][:, 0, 1:, :-1]

    num_points = len(u100.coords["lat"])
    new_latitude = np.linspace(27.0, 72.0, num=num_points)
    da_u100 = u100.assign_coords(lat=new_latitude)
    da_v100 = v100.assign_coords(lat=new_latitude)

    w_s = wind_speed(da_u100, da_v100)

    path_clim = hpcprojdir + "/applications/energy_indicators/ws_clim_1991_2020_eur.nc"
    ds_clim = xr.open_dataset(path_clim)
    ds_clim.close()
    clim_10m = ds_clim["sfcWind"]

    # Approximate 100m wind speed from 10m wind speed using power law as ERA5-Land does not provide
    # 100m wind speed.
    clim_100m = clim_10m * (100 / 10) ** (0.143)

    ws_anom = wind_speed_anomalies(w_s, clim_100m, scale="daily")

    date = pd.to_datetime(w_s["time"].values[0])
    yyyy_mm_dd = date.strftime("%Y_%m_%d")

    output_file_path = os.path.join(out_path, f"{yyyy_mm_dd}_T00_00_ws100_anom.nc")

    ws_anom.attrs["history"] = history

    # reduce file size by converting to float32
    ws_anom = ws_anom.astype(np.float32)

    ws_anom.to_netcdf(path=output_file_path, mode="w")
    print("Wind speed anomalies have been produced and saved to: ", output_file_path)


# Capacity factor (class I)


def run_capacity_factor_i(
    iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path, mask=None
):
    """


    Parameters
    ----------
    iniyear : string
        initial year of the streamed data YYYY.
    inimonth : string
        initial month of the streamed data MM.
    iniday : string
        initial day of the streamed data DD.
    finyear : string
        final year of the streamed data YYYY.
    finmonth : string
        final month of the streamed data MM.
    finday : string
        final day of the streamed data DD.
    in_path : string
        root path where to get the data from.
    out_path : string
        path where the output data goes to.
    mask : str or None (None by default).
        Path to a netCDF file containing a land-sea mask. If provided, the mask will be applied.

    Returns
    -------
    None.

    """

    # Provide the data file name for all variables

    u100_file = (
        f"{iniyear}_{inimonth}_{iniday}_T00_00_to_{finyear}_{finmonth}_{finday}_"
        "T23_00_u_raw_data.nc"
    )
    v100_file = (
        f"{iniyear}_{inimonth}_{iniday}_T00_00_to_{finyear}_{finmonth}_{finday}_"
        "T23_00_v_raw_data.nc"
    )

    absolute_path_u100 = os.path.join(in_path, u100_file)
    absolute_path_v100 = os.path.join(in_path, v100_file)

    data_u100 = xr.open_dataset(absolute_path_u100)
    data_v100 = xr.open_dataset(absolute_path_v100)

    # time
    time = get_time_utc()

    # version
    version = get_application_version()

    message = (
        time + " ENERGY: capacity factor (I type) computed using the "
        f"energy_indicators application v{version}."
    )

    history = data_u100.attrs["history"] + data_v100.attrs["history"] + message

    # Import processing script.

    u100 = data_u100["u"][:, 0, :, :]
    v100 = data_v100["v"][:, 0, :, :]

    w_s = wind_speed(u100, v100)

    # Add mask if provided
    c_f = capacity_factor(w_s, iec_class="I", mask=mask)

    # Global attrs:
    c_f.attrs = {
        "resolution": data_u100.attrs["resolution"],
        "generation": data_u100.attrs["generation"],
        "activity": data_u100.attrs["activity"],
        "dataset": data_u100.attrs["dataset"],
        "stream": data_u100.attrs["stream"],
        "model": data_u100.attrs["model"],
        "experiment": data_u100.attrs["experiment"],
        "levtype": data_u100.attrs["levtype"],
        "expver": data_u100.attrs["expver"],
        "class": data_u100.attrs["class"],
        "type": data_u100.attrs["type"],
        "realization": data_u100.attrs["realization"],
    }

    # Add mask attribute to global attributes if mask was applied
    if mask is not None:
        c_f.attrs["mask"] = "yes"

    c_f.attrs["history"] = history

    date = pd.to_datetime(w_s["time"].values[0])
    yyyy_mm_dd = date.strftime("%Y_%m_%d")

    output_file_path = os.path.join(out_path, f"{yyyy_mm_dd}_T00_00_cf_I.nc")

    c_f.to_netcdf(
        path=output_file_path, mode="w"
    )
    print(
        "Capacity factor for turbine type 'I' has been produced and saved to: ",
        output_file_path,
    )


# Capacity factor (class II)


def run_capacity_factor_ii(
    iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path, mask=None
):
    """


    Parameters
    ----------
    iniyear : string
        initial year of the streamed data YYYY.
    inimonth : string
        initial month of the streamed data MM.
    iniday : string
        initial day of the streamed data DD.
    finyear : string
        final year of the streamed data YYYY.
    finmonth : string
        final month of the streamed data MM.
    finday : string
        final day of the streamed data DD.
    in_path : string
        root path where to get the data from.
    out_path : string
        path where the output data goes to.
    mask : str or None (None by default).
        Path to a netCDF file containing a land-sea mask. If provided, the mask will be applied.

    Returns
    -------
    None.

    """

    # Provide the data file name for all variables

    u100_file = (
        f"{iniyear}_{inimonth}_{iniday}_T00_00_to_{finyear}_{finmonth}_{finday}_"
        "T23_00_u_raw_data.nc"
    )
    v100_file = (
        f"{iniyear}_{inimonth}_{iniday}_T00_00_to_{finyear}_{finmonth}_{finday}_"
        "T23_00_v_raw_data.nc"
    )

    absolute_path_u100 = os.path.join(in_path, u100_file)
    absolute_path_v100 = os.path.join(in_path, v100_file)

    data_u100 = xr.open_dataset(absolute_path_u100)
    data_v100 = xr.open_dataset(absolute_path_v100)

    # time
    time = get_time_utc()

    # version
    version = get_application_version()

    message = (
        time + " ENERGY: capacity factor (II type) computed using the "
        f"energy_indicators application v{version}."
    )

    history = data_u100.attrs["history"] + data_v100.attrs["history"] + message
    # Import processing script.

    u100 = data_u100["u"][:, 0, :, :]
    v100 = data_v100["v"][:, 0, :, :]

    w_s = wind_speed(u100, v100)

    c_f = capacity_factor(w_s, iec_class="II", mask=mask)

    # Global attrs:
    c_f.attrs = {
        "resolution": data_u100.attrs["resolution"],
        "generation": data_u100.attrs["generation"],
        "activity": data_u100.attrs["activity"],
        "dataset": data_u100.attrs["dataset"],
        "stream": data_u100.attrs["stream"],
        "model": data_u100.attrs["model"],
        "experiment": data_u100.attrs["experiment"],
        "levtype": data_u100.attrs["levtype"],
        "expver": data_u100.attrs["expver"],
        "class": data_u100.attrs["class"],
        "type": data_u100.attrs["type"],
        "realization": data_u100.attrs["realization"],
    }

    # Add mask attribute to global attributes if mask was applied
    if mask is not None:
        c_f.attrs["mask"] = "yes"

    c_f.attrs["history"] = history

    date = pd.to_datetime(w_s["time"].values[0])
    yyyy_mm_dd = date.strftime("%Y_%m_%d")

    output_file_path = os.path.join(out_path, f"{yyyy_mm_dd}_T00_00_cf_II.nc")

    c_f.to_netcdf(
        path=output_file_path, mode="w"
    )
    print(
        "Capacity factor for turbine type 'II' has been produced and saved to: ",
        output_file_path,
    )


# Capacity factor (class III)


def run_capacity_factor_iii(
    iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path, mask=None
):
    """


    Parameters
    ----------
    iniyear : string
        initial year of the streamed data YYYY.
    inimonth : string
        initial month of the streamed data MM.
    iniday : string
        initial day of the streamed data DD.
    finyear : string
        final year of the streamed data YYYY.
    finmonth : string
        final month of the streamed data MM.
    finday : string
        final day of the streamed data DD.
    in_path : string
        root path where to get the data from.
    out_path : string
        path where the output data goes to.
    mask : str or None (None by default).
        Path to a netCDF file containing a land-sea mask. If provided, the mask will be applied.

    Returns
    -------
    None.

    """

    # Provide the data file name for all variables

    u100_file = (
        f"{iniyear}_{inimonth}_{iniday}_T00_00_to_{finyear}_{finmonth}_{finday}_"
        "T23_00_u_raw_data.nc"
    )
    v100_file = (
        f"{iniyear}_{inimonth}_{iniday}_T00_00_to_{finyear}_{finmonth}_{finday}_"
        "T23_00_v_raw_data.nc"
    )

    absolute_path_u100 = os.path.join(in_path, u100_file)
    absolute_path_v100 = os.path.join(in_path, v100_file)

    data_u100 = xr.open_dataset(absolute_path_u100)
    data_v100 = xr.open_dataset(absolute_path_v100)

    # time
    time = get_time_utc()

    # version
    version = get_application_version()

    message = (
        time + " ENERGY: capacity factor (III type) computed using the "
        f"energy_indicators application v{version}."
    )

    history = data_u100.attrs["history"] + data_v100.attrs["history"] + message
    # Import processing script.

    u100 = data_u100["u"][:, 0, :, :]
    v100 = data_v100["v"][:, 0, :, :]

    w_s = wind_speed(u100, v100)

    c_f = capacity_factor(w_s, iec_class="III", mask=mask)

    # Global attrs:
    c_f.attrs = {
        "resolution": data_u100.attrs["resolution"],
        "generation": data_u100.attrs["generation"],
        "activity": data_u100.attrs["activity"],
        "dataset": data_u100.attrs["dataset"],
        "stream": data_u100.attrs["stream"],
        "model": data_u100.attrs["model"],
        "experiment": data_u100.attrs["experiment"],
        "levtype": data_u100.attrs["levtype"],
        "expver": data_u100.attrs["expver"],
        "class": data_u100.attrs["class"],
        "type": data_u100.attrs["type"],
        "realization": data_u100.attrs["realization"],
    }

    # Add mask attribute to global attributes if mask was applied
    if mask is not None:
        c_f.attrs["mask"] = "yes"

    c_f.attrs["history"] = history

    date = pd.to_datetime(w_s["time"].values[0])
    yyyy_mm_dd = date.strftime("%Y_%m_%d")

    output_file_path = os.path.join(out_path, f"{yyyy_mm_dd}_T00_00_cf_III.nc")

    c_f.to_netcdf(
        path=output_file_path, mode="w"
    )
    print(
        "Capacity factor for turbine type 'III' has been produced and saved to: ",
        output_file_path,
    )


# Capacity factor (class S)


def run_capacity_factor_s(
    iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path, mask=None
):
    """


    Parameters
    ----------
    iniyear : string
        initial year of the streamed data YYYY.
    inimonth : string
        initial month of the streamed data MM.
    iniday : string
        initial day of the streamed data DD.
    finyear : string
        final year of the streamed data YYYY.
    finmonth : string
        final month of the streamed data MM.
    finday : string
        final day of the streamed data DD.
    in_path : string
        root path where to get the data from.
    out_path : string
        path where the output data goes to.
    mask : str or None (None by default).
        Path to a netCDF file containing a land-sea mask. If provided, the mask will be applied.

    Returns
    -------
    None.

    """

    # Provide the data file name for all variables

    u100_file = (
        f"{iniyear}_{inimonth}_{iniday}_T00_00_to_{finyear}_{finmonth}_{finday}_"
        "T23_00_u_raw_data.nc"
    )
    v100_file = (
        f"{iniyear}_{inimonth}_{iniday}_T00_00_to_{finyear}_{finmonth}_{finday}_"
        "T23_00_v_raw_data.nc"
    )

    absolute_path_u100 = os.path.join(in_path, u100_file)
    absolute_path_v100 = os.path.join(in_path, v100_file)

    data_u100 = xr.open_dataset(absolute_path_u100)
    data_v100 = xr.open_dataset(absolute_path_v100)

    # time
    time = get_time_utc()

    # version
    version = get_application_version()

    message = (
        time + " ENERGY: capacity factor (S type) computed using the "
        f" energy_indicators application v{version}."
    )

    history = data_u100.attrs["history"] + data_v100.attrs["history"] + message

    # Import processing script.

    u100 = data_u100["u"][:, 0, :, :]
    v100 = data_v100["v"][:, 0, :, :]

    w_s = wind_speed(u100, v100)

    c_f = capacity_factor(w_s, iec_class="S", mask=mask)

    # Global attrs:
    c_f.attrs = {
        "resolution": data_u100.attrs["resolution"],
        "generation": data_u100.attrs["generation"],
        "activity": data_u100.attrs["activity"],
        "dataset": data_u100.attrs["dataset"],
        "stream": data_u100.attrs["stream"],
        "model": data_u100.attrs["model"],
        "experiment": data_u100.attrs["experiment"],
        "levtype": data_u100.attrs["levtype"],
        "expver": data_u100.attrs["expver"],
        "class": data_u100.attrs["class"],
        "type": data_u100.attrs["type"],
        "realization": data_u100.attrs["realization"],
    }

    # Add mask attribute to global attributes if mask was applied
    if mask is not None:
        c_f.attrs["mask"] = "yes"

    c_f.attrs["history"] = history

    date = pd.to_datetime(w_s["time"].values[0])
    yyyy_mm_dd = date.strftime("%Y_%m_%d")

    output_file_path = os.path.join(out_path, f"{yyyy_mm_dd}_T00_00_cf_S.nc")

    c_f.to_netcdf(
        path=output_file_path, mode="w"
    )
    print(
        "Capacity factor for turbine type 'S' has been produced and saved to: ",
        output_file_path,
    )


# Cooling degree days (CDD)


def run_cdd(iniyear, inimonth, iniday, in_path, out_path, mask=None):
    """


    Parameters
    ----------
    iniyear : string
        initial year of the streamed data YYYY.
    inimonth : string
        initial month of the streamed data MM.
    iniday : string
        initial day of the streamed data DD.
    in_path : string
        root path where to get the data from.
    out_path : string
        path where the output data goes to.
    mask : str or None (None by default).
        Path to a netCDF file containing a land-sea mask. Note: this
        parameter is currently not forwarded to the underlying
        computation, masking has no effect.

    Returns
    -------
    None.

    """

    # Provide the data file name for all variables

    t_file = f"{iniyear}_{inimonth}_{iniday}_2t_timestep_60_daily_mean.nc"
    tmax_file = f"{iniyear}_{inimonth}_{iniday}_2t_timestep_60_daily_max.nc"
    tmin_file = f"{iniyear}_{inimonth}_{iniday}_2t_timestep_60_daily_min.nc"

    absolute_path = os.path.join(in_path, t_file)
    absolute_path_max = os.path.join(in_path, tmax_file)
    absolute_path_min = os.path.join(in_path, tmin_file)

    data = xr.open_dataset(absolute_path)
    data_max = xr.open_dataset(absolute_path_max)
    data_min = xr.open_dataset(absolute_path_min)

    # time
    time = get_time_utc()

    # version
    version = get_application_version()

    message = (
        time + " ENERGY: cooling degree days computed using the "
        f" energy_indicators application v{version}."
    )

    history = (
        data.attrs["history"]
        + data_max.attrs["history"]
        + data_min.attrs["history"]
        + message
    )
    # Import processing script.

    data = data["2t"]
    data_max = data_max["2t"]
    data_min = data_min["2t"]
    t_m = convert_temperature(data, unit="C")
    t_x = convert_temperature(data_max, unit="C")
    t_n = convert_temperature(data_min, unit="C")

    cdd = cooling_degree_days(t_m, t_x, t_n, base=22.0)

    # Global attrs:
    cdd.attrs = {
        "resolution": data_max.attrs["resolution"],
        "generation": data_max.attrs["generation"],
        "activity": data_max.attrs["activity"],
        "dataset": data_max.attrs["dataset"],
        "stream": data_max.attrs["stream"],
        "model": data_max.attrs["model"],
        "experiment": data_max.attrs["experiment"],
        "levtype": data_max.attrs["levtype"],
        "expver": data_max.attrs["expver"],
        "class": data_max.attrs["class"],
        "type": data_max.attrs["type"],
        "realization": data_max.attrs["realization"],
    }

    # Add mask attribute to global attributes if mask was applied
    if mask is not None:
        cdd.attrs["mask"] = "yes"

    cdd.attrs["history"] = history

    date = pd.to_datetime(data["time"].values[0])
    yyyy_mm_dd = date.strftime("%Y_%m_%d")

    output_file_path = os.path.join(out_path, f"{yyyy_mm_dd}_T00_00_cdd.nc")

    # reduce file size by packing to int16 (scale_factor / add_offset)
    cdd.to_netcdf(
        path=output_file_path, mode="w", encoding=_packed_int16_encoding(cdd)
    )
    print("Cooling degree days have been produced and saved to: ", output_file_path)


# Heating degree days (HDD)


def run_hdd(iniyear, inimonth, iniday, in_path, out_path, mask=None):
    """


    Parameters
    ----------
    iniyear : string
        initial year of the streamed data YYYY.
    inimonth : string
        initial month of the streamed data MM.
    iniday : string
        initial day of the streamed data DD.
    in_path : string
        root path where to get the data from.
    out_path : string
        path where the output data goes to.
    mask : str or None (None by default).
        Path to a netCDF file containing a land-sea mask. Note: this
        parameter is currently not forwarded to the underlying
        computation, masking has no effect.

    Returns
    -------
    None.

    """

    # Provide the data file name for all variables
    t_file = f"{iniyear}_{inimonth}_{iniday}_2t_timestep_60_daily_mean.nc"
    tmax_file = f"{iniyear}_{inimonth}_{iniday}_2t_timestep_60_daily_max.nc"
    tmin_file = f"{iniyear}_{inimonth}_{iniday}_2t_timestep_60_daily_min.nc"

    absolute_path = os.path.join(in_path, t_file)
    absolute_path_max = os.path.join(in_path, tmax_file)
    absolute_path_min = os.path.join(in_path, tmin_file)

    data = xr.open_dataset(absolute_path)
    data_max = xr.open_dataset(absolute_path_max)
    data_min = xr.open_dataset(absolute_path_min)

    # time
    time = get_time_utc()

    # version
    version = get_application_version()

    message = (
        time + f" ENERGY: heating degree days computed using the "
        f"energy_indicators application v{version}."
    )

    history = (
        data.attrs["history"]
        + data_max.attrs["history"]
        + data_min.attrs["history"]
        + message
    )

    # Import processing script.

    data = data["2t"]
    data_max = data_max["2t"]
    data_min = data_min["2t"]
    t_m = convert_temperature(data, unit="C")
    t_x = convert_temperature(data_max, unit="C")
    t_n = convert_temperature(data_min, unit="C")

    hdd = heating_degree_days(t_m, t_x, t_n, base=15.5)
    hdd.attrs["history"] = history

    # Global attrs:
    hdd.attrs = {
        "resolution": data_max.attrs["resolution"],
        "generation": data_max.attrs["generation"],
        "activity": data_max.attrs["activity"],
        "dataset": data_max.attrs["dataset"],
        "stream": data_max.attrs["stream"],
        "model": data_max.attrs["model"],
        "experiment": data_max.attrs["experiment"],
        "levtype": data_max.attrs["levtype"],
        "expver": data_max.attrs["expver"],
        "class": data_max.attrs["class"],
        "type": data_max.attrs["type"],
        "realization": data_max.attrs["realization"],
    }

    # Add mask attribute to global attributes if mask was applied
    if mask is not None:
        hdd.attrs["mask"] = "yes"

    hdd.attrs["history"] = history

    date = pd.to_datetime(data["time"].values[0])
    yyyy_mm_dd = date.strftime("%Y_%m_%d")

    output_file_path = os.path.join(out_path, f"{yyyy_mm_dd}_T00_00_hdd.nc")

    # reduce file size by packing to int16 (scale_factor / add_offset)
    hdd.to_netcdf(
        path=output_file_path, mode="w", encoding=_packed_int16_encoding(hdd)
    )
    print("Heating degree days have been produced and saved to: ", output_file_path)


# High wind events


def run_wind_producing_regimes(
    iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path, mask=None
):
    """


    Parameters
    ----------
    iniyear : string
        initial year of the streamed data YYYY.
    inimonth : string
        initial month of the streamed data MM.
    iniday : string
        initial day of the streamed data DD.
    finyear : string
        final year of the streamed data YYYY.
    finmonth : string
        final month of the streamed data MM.
    finday : string
        final day of the streamed data DD.
    in_path : string
        root path where to get the data from.
    out_path : string
        path where the output data goes to.
    mask : str or None (None by default).
        Path to a netCDF file containing a land-sea mask. Note: this
        parameter is currently not forwarded to the underlying
        computation, masking has no effect.

    Returns
    -------
    Writes a netcdf file with low and high wind events and producing time as variables.

    """

    # Provide the data file name for all variables

    u100_file = (
        f"{iniyear}_{inimonth}_{iniday}_T00_00_to_{finyear}_{finmonth}_{finday}_"
        "T23_00_u_raw_data.nc"
    )
    v100_file = (
        f"{iniyear}_{inimonth}_{iniday}_T00_00_to_{finyear}_{finmonth}_{finday}_"
        "T23_00_v_raw_data.nc"
    )

    absolute_path_u100 = os.path.join(in_path, u100_file)
    absolute_path_v100 = os.path.join(in_path, v100_file)

    data_u100 = xr.open_dataset(absolute_path_u100)
    data_v100 = xr.open_dataset(absolute_path_v100)

    # time
    time = get_time_utc()

    # version
    version = get_application_version()

    message = (
        time + " ENERGY: high wind events computed using the "
        f"energy_indicators application v{version}."
    )

    history = data_u100.attrs["history"] + data_v100.attrs["history"] + message

    # Import processing script.

    u100 = data_u100["u"][:, 0, :, :]
    v100 = data_v100["v"][:, 0, :, :]

    w_s = wind_speed(u100, v100)

    hwe = high_wind_events(w_s, threshold=25.0)
    lwe = low_wind_events(w_s, threshold=3.0)

    # create a new dataset to store the producing time
    producing_time = xr.Dataset()
    
    total_timesteps = xr.full_like(lwe["lwe"], fill_value=w_s.sizes['time'])    
    producing_time['producing_time'] =  total_timesteps - (lwe['lwe'] + hwe['hwe'])
    producing_time['producing_time'].attrs['shortname'] = 'producing_time'
    producing_time['producing_time'].attrs['longname'] = 'Number of timesteps with wind speed between 3 and 25 m/s'
    producing_time['producing_time'].attrs['units'] = '-'

    wpr = xr.Dataset(data_vars={"lwe": lwe["lwe"], "hwe": hwe["hwe"], "producing_time": producing_time["producing_time"]}, coords=w_s.coords)

    # Global attrs:
    wpr.attrs = {
        "resolution": data_u100.attrs["resolution"],
        "generation": data_u100.attrs["generation"],
        "activity": data_u100.attrs["activity"],
        "dataset": data_u100.attrs["dataset"],
        "stream": data_u100.attrs["stream"],
        "model": data_u100.attrs["model"],
        "experiment": data_u100.attrs["experiment"],
        "levtype": data_u100.attrs["levtype"],
        "expver": data_u100.attrs["expver"],
        "class": data_u100.attrs["class"],
        "type": data_u100.attrs["type"],
        "realization": data_u100.attrs["realization"],
    }

    # Add mask attribute to global attributes if mask was applied
    if mask is not None:
        wpr.attrs["mask"] = "yes"

    wpr.attrs["history"] = history

    date = pd.to_datetime(w_s["time"].values[0])
    yyyy_mm_dd = date.strftime("%Y_%m_%d")

    output_file_path = os.path.join(out_path, f"{yyyy_mm_dd}_T00_00_wpr.nc")
    
    # reduce file size by converting to int16
    wpr['lwe']=lwe['lwe'].astype(np.int16)
    wpr['hwe']=hwe['hwe'].astype(np.int16)
    wpr['producing_time']=producing_time['producing_time'].astype(np.int16)
        
    wpr.to_netcdf(path=output_file_path, mode="w")
    print("Wind producing regimes have been produced and saved to: ", output_file_path)

# PV potential


def run_pv_potential(
    iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path, mask=None
):
    """
    Compute PV potential (PVP) from hourly temperature (2t), shortwave radiation
    (rsds/avg_sdswrf in W/m2 or ssrd in J/m2), and wind speed (10si).
    Computes hourly PVP and aggregates it to a daily mean.
    Writes a NetCDF and returns None.

    Parameters
    ----------

    iniyear : string
        initial year of the streamed data YYYY.
    inimonth : string
        initial month of the streamed data MM.
    iniday : string
        initial day of the streamed data DD.
    finyear : string
        final year of the streamed data YYYY.
    finmonth : string
        final month of the streamed data MM.
    finday : string
        final day of the streamed data DD.
    in_path : string
        root path where to get the data from.
    out_path : string
        path where the output data goes to.
    mask : str or None (None by default).
        Path to a netCDF file containing a land-sea mask. Note: this
        parameter is currently not forwarded to the underlying
        computation, masking has no effect.

    """

    # --- Filenames (hourly raw), mirroring existing pattern ---
    t2_file = (
        f"{iniyear}_{inimonth}_{iniday}_T00_00_to_"
        f"{finyear}_{finmonth}_{finday}_T23_00_2t_raw_data.nc"
    )
    avg_sdswrf_file = (
        f"{iniyear}_{inimonth}_{iniday}_T00_00_to_"
        f"{finyear}_{finmonth}_{finday}_T23_00_avg_sdswrf_raw_data.nc"
    )
    ws_file = (
        f"{iniyear}_{inimonth}_{iniday}_T00_00_to_"
        f"{finyear}_{finmonth}_{finday}_T23_00_10si_raw_data.nc"
    )

    # --- Absolute paths ---
    p_t2 = os.path.join(in_path, t2_file)
    p_avg_sdswrf = os.path.join(in_path, avg_sdswrf_file)
    p_ws = os.path.join(in_path, ws_file)

    # --- Open mandatory datasets ---
    data_t2 = xr.open_dataset(p_t2)
    data_g = xr.open_dataset(p_avg_sdswrf)
    data_ws = xr.open_dataset(p_ws)

    # --- Metadata (same style as other run_* functions) ---
    time = get_time_utc()
    version = get_application_version()
    message = (
        time + " ENERGY: PV potential computed using the "
        f"energy_indicators application v{version}."
    )

    history = (
        data_t2.attrs["history"]
        + data_g.attrs["history"]
        + data_ws.attrs["history"]
        + message
    )

    # --- Prepare inputs: 2t in °C and wind speed magnitude ---
    t2k = data_t2["2t"]  # Kelvin
    g = data_g["avg_sdswrf"]
    w_s = data_ws["10si"]

    t2c = convert_temperature(t2k, unit="C")

    # Compute hourly PVP
    pvp = pv_pot(t2c, g, w_s)

    # Aggregate hourly PVP to a daily mean
    daily_pvp = daily_pv_potential(pvp)

    # --- Global attrs (mirror pattern in other run_* functions) ---
    daily_pvp.attrs = {
        "resolution": t2k.attrs["resolution"],
        "generation": t2k.attrs["generation"],
        "activity": t2k.attrs["activity"],
        "dataset": t2k.attrs["dataset"],
        "stream": t2k.attrs["stream"],
        "model": t2k.attrs["model"],
        "experiment": t2k.attrs["experiment"],
        "levtype": t2k.attrs["levtype"],
        "expver": t2k.attrs["expver"],
        "class": t2k.attrs["class"],
        "type": t2k.attrs["type"],
        "realization": t2k.attrs["realization"],
    }

    # Add mask attribute to global attributes if mask was applied
    # TODO: the mask is neve applied in this function
    if mask is not None:
        daily_pvp.attrs["mask"] = "yes"

    daily_pvp.attrs["history"] = history

    # --- Output file (consistent naming) ---
    date = pd.to_datetime(daily_pvp["time"].values[0])
    yyyy_mm_dd = date.strftime("%Y_%m_%d")
    output_file_path = os.path.join(out_path, f"{yyyy_mm_dd}_T00_00_pvp.nc")

    # reduce file size by packing to int16 (scale_factor / add_offset)
    daily_pvp.to_netcdf(
        path=output_file_path, mode="w", encoding=_packed_int16_encoding(daily_pvp, scale_factor=0.001)
    )
    print("PV potential has been produced and saved to:", output_file_path)


# Capacity factor histogram (OPA)


def run_capacity_factor_histogram_opa(
    iniyear,
    inimonth,
    iniday,
    in_path,
    out_path,
    mask=None,
    cf_type="I",
    nworkers=1,
):
    """


    Parameters
    ----------
    iniyear : string
        initial year of the streamed data YYYY.
    inimonth : string
        initial month of the streamed data MM.
    iniday : string
        initial day of the streamed data DD.
    in_path : string
        root path where to get the data from.
    out_path:
        path where the result goes.
    cf_type : string
        type of capacity factor: I, II, III, S.
    nworkers : int
        number of workers to use for parallel processing (if applicable).
    mask : str or None (None by default).
        Path to a netCDF file containing a land-sea mask. If provided, the mask will be applied.

    Returns
    -------
    None.

    """
    # Provide the data file name for all variables
    cf_file = f"{iniyear}_{inimonth}_{iniday}_T00_00_cf_{cf_type}.nc"

    absolute_path_cf = os.path.join(in_path, cf_file)

    data_cf = xr.open_dataset(absolute_path_cf)

    cf = data_cf[f"cf_{cf_type.lower()}"]

    # Global attrs:
    cf.attrs = data_cf.attrs.copy()

    out_path=in_path # this is to temporarilly overcome the fact that there is no out pathe in capacity_factor_histogram_opa

    print(f"mask in run_energy_indicators python repo {mask}", flush=True)
    capacity_factor_histogram_opa(
        cf, in_path, mask=mask, iec_class=f"{cf_type.lower()}", nworkers=nworkers
    )


def run_wind_power_density(
    iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path, mask=None
):
    """
    Parameters
    ----------
    iniyear : string
        initial year of the streamed data YYYY.
    inimonth : string
        initial month of the streamed data MM.
    iniday : string
        initial day of the streamed data DD.
    finyear : string
        final year of the streamed data YYYY.
    finmonth : string
        final month of the streamed data MM.
    finday : string
        final day of the streamed data DD.
    in_path : string
        root path where to get the data from.
    out_path : string
        path where the output data goes to.
    mask : str or None (None by default).
        Path to a netCDF file containing a land-sea mask. If provided, the mask will be applied.

    Returns
    -------
    None.
    """
    # Provide the data file name for all variables
    u100_file = (
        f"{iniyear}_{inimonth}_{iniday}_T00_00_to_{finyear}_{finmonth}_{finday}_"
        "T23_00_u_raw_data.nc"
    )
    v100_file = (
        f"{iniyear}_{inimonth}_{iniday}_T00_00_to_{finyear}_{finmonth}_{finday}_"
        "T23_00_v_raw_data.nc"
    )

    absolute_path_u100 = os.path.join(in_path, u100_file)
    absolute_path_v100 = os.path.join(in_path, v100_file)

    data_u100 = xr.open_dataset(absolute_path_u100)
    data_v100 = xr.open_dataset(absolute_path_v100)

    # time
    time = get_time_utc()

    # version
    version = get_application_version()

    message = (
        time + " ENERGY: wind power density computed using the "
        f"energy_indicators application v{version}."
    )

    history = data_u100.attrs["history"] + data_v100.attrs["history"] + message

    u100 = data_u100["u"][:, 0, :, :]
    v100 = data_v100["v"][:, 0, :, :]

    w_s = wind_speed(u100, v100)

    # Add mask if provided
    wpd = wind_power_density(w_s, mask=mask)

    # Aggregate hourly WPD to a daily mean
    daily_wpd = daily_wind_power_density(wpd)

    # Global attrs:
    daily_wpd.attrs = {
        "resolution": data_u100.attrs["resolution"],
        "generation": data_u100.attrs["generation"],
        "activity": data_u100.attrs["activity"],
        "dataset": data_u100.attrs["dataset"],
        "stream": data_u100.attrs["stream"],
        "model": data_u100.attrs["model"],
        "experiment": data_u100.attrs["experiment"],
        "levtype": data_u100.attrs["levtype"],
        "expver": data_u100.attrs["expver"],
        "class": data_u100.attrs["class"],
        "type": data_u100.attrs["type"],
        "realization": data_u100.attrs["realization"],
    }

    # Add mask attribute to global attributes if mask was applied
    if mask is not None:
        daily_wpd.attrs["mask"] = "yes"

    daily_wpd.attrs["history"] = history

    date = pd.to_datetime(daily_wpd["time"].values[0])
    yyyy_mm_dd = date.strftime("%Y_%m_%d")

    output_file_path = os.path.join(out_path, f"{yyyy_mm_dd}_T00_00_wpd.nc")

    # reduce file size by converting to float32
    daily_wpd = daily_wpd.astype(np.float32)

    daily_wpd.to_netcdf(path=output_file_path, mode="w")
    print("Wind power density has been produced and saved to: ", output_file_path)


# NUTS spatial aggregation


def run_nuts_aggregation(
    iniyear,
    inimonth,
    iniday,
    indicator,
    in_path,
    out_path,
    region_mask_file,
    lat_weights_file=None,
    stat="mean",
    region_level=0,
    n_jobs=1,
):
    """
    Spatially aggregate an already-computed indicator field to country-
    level (or other) regions.

    Unlike the other run_* functions, this does NOT read raw met variables -
    it reads the indicator NetCDF already produced by the corresponding
    run_* function (e.g. run_cdd, run_wind_power_density), following the
    same "{yyyy_mm_dd}_T00_00_{indicator}.nc" naming convention those
    functions write to. This keeps aggregation decoupled from how each
    indicator was computed - by the time this runs, the indicator is
    already a materialised file.
        Initial year of the indicator file to aggregate, YYYY.
    inimonth : string
        Initial month of the indicator file to aggregate, MM.
    iniday : string
        Initial day of the indicator file to aggregate, DD.
    indicator : string
        Short name of the indicator, matching its filename suffix (e.g.
        "cdd", "hdd", "wpd", "cf_i"). Used both to locate the input file
        and to label the output.
    in_path : string
        Path to the directory containing the already-computed indicator
        NetCDF (i.e. out_path of the corresponding run_* function).
    out_path : string
        Path where the aggregated NetCDF goes.
    region_mask_file : string
        Path to a precomputed fractional-coverage region mask NetCDF (see
        aggregation.build_region_mask / get_or_build_region_mask).
    lat_weights_file : string or None
        Optional precomputed weights file (e.g. PECD/CDS, for validation runs).
        If None, cosine-latitude weights are computed from the region mask's
        grid for regular lat-lon grids.
    stat : string
        One of "mean", "sum", "min", "max", "std". Default "mean" - the
        others are supported (via aggregate_region) but not expected to
        see regular use here; exposed mainly for exploratory/diagnostic use.
    region_level : int
        Region level of region_mask_file (e.g. 0 or 2 for NUTS0/NUTS2).
        Only used for output file naming - does not affect the
        aggregation itself, which is agnostic to which region set
        region_mask_file encodes.
    n_jobs : int
        Worker processes passed through to aggregate_region. Default 1
        (sequential). At native grid resolution the mask can be tens of
        GB, so this step is memory-bandwidth bound; splitting the region
        dimension across cores gives a meaningful speedup. Not automatic,
        set explicitly to match the cores available when calling this
        from a batch job.

    Returns
    -------
    None. Writes a NetCDF (time, region) to out_path.
    """
    input_file = os.path.join(
        in_path, f"{iniyear}_{inimonth}_{iniday}_T00_00_{indicator}.nc"
    )
    data = xr.open_dataset(input_file)
    var_name = indicator if indicator in data.data_vars else list(data.data_vars)[0]
    data_var = data[var_name]

    region_mask = load_region_mask(region_mask_file)
    if lat_weights_file is not None:
        lat_weights = load_lat_weights(lat_weights_file)
    else:
        lat_weights = build_lat_weights(region_mask.lat.values, region_mask.lon.values)

    aggregated = aggregate_region(data_var, region_mask, lat_weights, stat=stat, n_jobs=n_jobs)
    aggregated_ds = aggregated.to_dataset()

    time = get_time_utc()
    version = get_application_version()
    message = (
        f"{time} ENERGY: {indicator} aggregated to region level "
        f"{region_level} using the energy_indicators application v{version}."
    )
    aggregated_ds.attrs = dict(data.attrs)
    aggregated_ds.attrs["history"] = data.attrs.get("history", "") + message
    aggregated_ds.attrs["aggregation_stat"] = stat
    aggregated_ds.attrs["region_level"] = region_level

    date = pd.to_datetime(data_var["time"].values[0])
    yyyy_mm_dd = date.strftime("%Y_%m_%d")
    output_file_path = os.path.join(
        out_path, f"{yyyy_mm_dd}_T00_00_{indicator}_nuts{region_level}.nc"
    )
    # reduce file size by converting to float32
    aggregated_ds = aggregated_ds.astype(np.float32)

    aggregated_ds.to_netcdf(path=output_file_path, mode="w")
    print(
        f"NUTS{region_level} aggregation of {indicator} has been produced and "
        f"saved to: {output_file_path}"
    )

# Daily solar cell temperature indicators

def run_daily_cell_temp(
    iniyear,
    inimonth,
    iniday,
    finyear,
    finmonth,
    finday,
    in_path,
    out_path,
    mask=None,
):
    """
    Compute daily average and daily maximum solar cell temperature from
    hourly temperature (2t), shortwave radiation (avg_sdswrf), and
    wind speed (10si).

    Parameters
    ----------
    iniyear : string
        initial year of the streamed data YYYY.
    inimonth : string
        initial month of the streamed data MM.
    iniday : string
        initial day of the streamed data DD.
    finyear : string
        final year of the streamed data YYYY.
    finmonth : string
        final month of the streamed data MM.
    finday : string
        final day of the streamed data DD.
    in_path : string
        root path where to get the data from.
    out_path : string
        path where the output data goes to.

    Returns
    -------
    None.

    """

    # Provide the data file name for all variables

    t2_file = (
        f"{iniyear}_{inimonth}_{iniday}_T00_00_to_"
        f"{finyear}_{finmonth}_{finday}_T23_00_2t_raw_data.nc"
    )

    avg_sdswrf_file = (
        f"{iniyear}_{inimonth}_{iniday}_T00_00_to_"
        f"{finyear}_{finmonth}_{finday}_T23_00_avg_sdswrf_raw_data.nc"
    )

    ws_file = (
        f"{iniyear}_{inimonth}_{iniday}_T00_00_to_"
        f"{finyear}_{finmonth}_{finday}_T23_00_10si_raw_data.nc"
    )

    absolute_path_t2 = os.path.join(in_path, t2_file)
    absolute_path_g = os.path.join(in_path, avg_sdswrf_file)
    absolute_path_ws = os.path.join(in_path, ws_file)

    data_t2 = xr.open_dataset(absolute_path_t2)
    data_g = xr.open_dataset(absolute_path_g)
    data_ws = xr.open_dataset(absolute_path_ws)

    # time
    time = get_time_utc()

    # version
    version = get_application_version()

    message = (
        time
        + " ENERGY: daily solar cell temperature indicators computed using the "
        f"energy_indicators application v{version}."
    )

    history = (
        data_t2.attrs["history"]
        + data_g.attrs["history"]
        + data_ws.attrs["history"]
        + message
    )

    # Import processing script.

    t2 = data_t2["2t"]
    g = data_g["avg_sdswrf"]
    ws = data_ws["10si"]

    # Convert temperature from Kelvin to degree Celsius
    t2 = convert_temperature(t2, unit="C")

    # Compute hourly cell temperature
    celltemp = cell_temp(t2, g, ws)

    # Compute daily indicators
    daily_ct = daily_cell_temp(celltemp)

    # Global attrs:
    daily_ct.attrs.update(
        {
            "resolution": data_t2.attrs["resolution"],
            "generation": data_t2.attrs["generation"],
            "activity": data_t2.attrs["activity"],
            "dataset": data_t2.attrs["dataset"],
            "stream": data_t2.attrs["stream"],
            "model": data_t2.attrs["model"],
            "experiment": data_t2.attrs["experiment"],
            "levtype": data_t2.attrs["levtype"],
            "expver": data_t2.attrs["expver"],
            "class": data_t2.attrs["class"],
            "type": data_t2.attrs["type"],
            "realization": data_t2.attrs["realization"],
        }
    )

    # Add mask attribute to global attributes if mask was applied
    # TODO: the mask is never applied in this function
    if mask is not None:
        daily_ct.attrs["mask"] = "yes"

    daily_ct.attrs["history"] = history

    date = pd.to_datetime(daily_ct["time"].values[0])
    yyyy_mm_dd = date.strftime("%Y_%m_%d")

    output_file_path = os.path.join(
        out_path,
        f"{yyyy_mm_dd}_T00_00_daily_cell_temp.nc",
    )

    # reduce file size by packing to int16 (scale_factor / add_offset)
    daily_ct.to_netcdf(
        path=output_file_path, mode="w", encoding=_packed_int16_encoding(daily_ct)
    )

    print(
        "Daily solar cell temperature indicators have been produced and saved to:",
        output_file_path,
    )
