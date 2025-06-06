#!/usr/bin/env python3
"""
Definition of the runs cripts to be run in the workflow runscript
"""

# Load libraries
import importlib
import os
from datetime import datetime, timezone

import numpy as np
import xarray as xr
import pandas as pd

from energy_onshore.core import wind_speed, convert_temperature
from energy_onshore.demand import (
    heating_degree_days,
    cooling_degree_days,
)

from energy_onshore.wind import (
    wind_speed_anomalies,
    capacity_factor,
    high_wind_events,
    low_wind_events,
)
# get time UTC

def get_time_utc():  # add pytest
    """
    Parameters
    ----------
    -

    Returns
    -------
    Current date time.
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
    package version.
    """
    version=importlib.metadata.version('energy_onshore')
    
    return version

# Wind speed anomalies


def run_wind_speed_anomalies(iniyear, inimonth, iniday, in_path, out_path, hpcprojdir):
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
        time + f" ENERGY: wind speed anomalies computed using the" 
        "energy_onshore application v{version}."
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

    path_clim = hpcprojdir + "/applications/energy_onshore/ws_clim_1991_2020_eur.nc"
    ds_clim = xr.open_dataset(path_clim)
    ds_clim.close()
    clim_10m = ds_clim["sfcWind"]

    # Approximate 100m wind speed from 10m wind speed using power law as ERA5-Land does not provide
    # 100m wind speed.
    clim_100m = clim_10m * (100 / 10) ** (0.143)

    ws_anom = wind_speed_anomalies(w_s, clim_100m, scale="daily")

    date = pd.to_datetime(w_s['time'].values[0])
    YYYY_MM_DD = date.strftime('%Y_%m_%d')

    output_file_path = os.path.join(out_path, f"{YYYY_MM_DD}_T00_00_ws100_anom.nc")

    ws_anom.attrs["history"] = history

    ws_anom.to_netcdf(path=output_file_path, mode="w")
    print("Wind speed anomalies have been produced and saved to: ", output_file_path)

# Capacity factor (class I)


def run_capacity_factor_i(
    iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path
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
        time + " ENERGY: capacity factor (I type) computed using the"
        f"energy_onshore application v{version}."
    )

    history = data_u100.attrs["history"] + data_v100.attrs["history"] + message

    # Import processing script.

    u100 = data_u100["u"][:, 0, :, :]
    v100 = data_v100["v"][:, 0, :, :]

    w_s = wind_speed(u100, v100)

    c_f = capacity_factor(w_s, iec_class="I")

    # Global attrs:
    c_f.attrs = {"resolution": data_u100.attrs["resolution"],
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
            "realization": data_u100.attrs["realization"]
            }

    c_f.attrs["history"] = history
    
    date = pd.to_datetime(w_s['time'].values[0])
    YYYY_MM_DD = date.strftime('%Y_%m_%d')
    
    output_file_path = os.path.join(out_path, f"{YYYY_MM_DD}_T00_00_cf_I.nc")

    c_f.to_netcdf(path=output_file_path, mode="w")
    print("Capacity factor for turbine type 'I' has been produced and saved to: ", output_file_path)


# Capacity factor (class II)


def run_capacity_factor_ii(
    iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path
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
        time + " ENERGY: capacity factor (II type) computed using the" 
        f"energy_onshore application v{version}."
    )

    history = data_u100.attrs["history"] + data_v100.attrs["history"] + message
    # Import processing script.

    u100 = data_u100["u"][:, 0, :, :]
    v100 = data_v100["v"][:, 0, :, :]

    w_s = wind_speed(u100, v100)

    c_f = capacity_factor(w_s, iec_class="II")

    # Global attrs:
    c_f.attrs = {"resolution": data_u100.attrs["resolution"],
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
            "realization": data_u100.attrs["realization"]
            }


    c_f.attrs["history"] = history

    date = pd.to_datetime(w_s['time'].values[0])
    YYYY_MM_DD = date.strftime('%Y_%m_%d')
    
    output_file_path = os.path.join(out_path, f"{YYYY_MM_DD}_T00_00_cf_II.nc")

    c_f.to_netcdf(path=output_file_path, mode="w")
    print("Capacity factor for turbine type 'II' has been produced and saved to: ", output_file_path)


# Capacity factor (class III)


def run_capacity_factor_iii(
    iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path
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
        time + " ENERGY: capacity factor (III type) computed using the"
        f"energy_onshore application v{version}."
    )

    history = data_u100.attrs["history"] + data_v100.attrs["history"] + message
    # Import processing script.

    u100 = data_u100["u"][:, 0, :, :]
    v100 = data_v100["v"][:, 0, :, :]

    w_s = wind_speed(u100, v100)

    c_f = capacity_factor(w_s, iec_class="III")

    # Global attrs:
    c_f.attrs = {"resolution": data_u100.attrs["resolution"],
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
            "realization": data_u100.attrs["realization"]
            }

    c_f.attrs["history"] = history

    date = pd.to_datetime(w_s['time'].values[0])
    YYYY_MM_DD = date.strftime('%Y_%m_%d')
    
    output_file_path = os.path.join(out_path, f"{YYYY_MM_DD}_T00_00_cf_III.nc")

    c_f.to_netcdf(path=output_file_path, mode="w")
    print("Capacity factor for turbine type 'III' has been produced and saved to: ", output_file_path)


# Capacity factor (class S)


def run_capacity_factor_s(
    iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path
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
        time + " ENERGY: capacity factor (S type) computed using the"
        f" energy_onshore application v{version}."
    )

    history = data_u100.attrs["history"] + data_v100.attrs["history"] + message

    # Import processing script.

    u100 = data_u100["u"][:, 0, :, :]
    v100 = data_v100["v"][:, 0, :, :]

    w_s = wind_speed(u100, v100)

    c_f = capacity_factor(w_s, iec_class="S")

    # Global attrs:
    c_f.attrs = {"resolution": data_u100.attrs["resolution"],
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
            "realization": data_u100.attrs["realization"]
            }

    c_f.attrs["history"] = history
    
    date = pd.to_datetime(w_s['time'].values[0])
    YYYY_MM_DD = date.strftime('%Y_%m_%d')
    
    output_file_path = os.path.join(out_path, f"{YYYY_MM_DD}_T00_00_cf_S.nc")

    c_f.to_netcdf(path=output_file_path, mode="w")
    print("Capacity factor for turbine type 'S' has been produced and saved to: ", output_file_path)


# Cooling degree days (CDD)


def run_cdd(iniyear, inimonth, iniday, in_path, out_path):
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
        f" energy_onshore application v{version}."
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

    cdd, cdd_acc = cooling_degree_days(t_m, t_x, t_n, base=22.0)

    # Global attrs:
    cdd.attrs = {"resolution": data_max.attrs["resolution"],
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
            "realization": data_max.attrs["realization"]
            }

    cdd.attrs["history"] = history
    cdd_acc.attrs["history"] = history

    cdd_acc.attrs = cdd.attrs

    date = pd.to_datetime(data['time'].values[0])
    YYYY_MM_DD = date.strftime('%Y_%m_%d')

    output_file_path = os.path.join(out_path, f"{YYYY_MM_DD}_T00_00_cdd.nc")
    output_file_path1 = os.path.join(out_path, f"{YYYY_MM_DD}_T00_00_cdd_acc.nc")

    cdd.to_netcdf(path=output_file_path, mode="w")
    cdd_acc.to_netcdf(path=output_file_path1, mode="w")
    print("Cooling degree days have been produced and saved to: ", output_file_path)

# Heating degree days (HDD)


def run_hdd(iniyear, inimonth, iniday, in_path, out_path):
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
        f"energy_onshore application v{version}."
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

    hdd, hdd_acc = heating_degree_days(t_m, t_x, t_n, base=15.5)
    hdd.attrs["history"] = history  # add hdd_acc output? --> it applies also for cdd

    # Global attrs:
    hdd.attrs = {"resolution": data_max.attrs["resolution"],
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
            "realization": data_max.attrs["realization"]
            }

    hdd_acc.attrs = hdd.attrs

    hdd.attrs["history"] = history
    hdd_acc.attrs["history"] = history


    date = pd.to_datetime(data['time'].values[0])
    YYYY_MM_DD = date.strftime('%Y_%m_%d')

    output_file_path = os.path.join(out_path, f"{YYYY_MM_DD}_T00_00_hdd.nc")
    output_file_path1 = os.path.join(out_path, f"{YYYY_MM_DD}_T00_00_hdd_acc.nc")

    hdd.to_netcdf(path=output_file_path, mode="w")
    hdd_acc.to_netcdf(path=output_file_path1, mode="w")
    print("Heating degree days have been produced and saved to: ", output_file_path)

# High wind events


def run_high_wind_events(
    iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path
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
        time + " ENERGY: high wind events computed using the "
        f"energy_onshore application v{version}."
    )

    history = data_u100.attrs["history"] + data_v100.attrs["history"] + message

    # Import processing script.

    u100 = data_u100["u"][:, 0, :, :]
    v100 = data_v100["v"][:, 0, :, :]

    w_s = wind_speed(u100, v100)

    hwe = high_wind_events(w_s, threshold=25.0)
    
    # Global attrs:
    hwe.attrs = {"resolution": data_u100.attrs["resolution"],
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
            "realization": data_u100.attrs["realization"]
            }
    
    hwe.attrs["history"] = history

    date = pd.to_datetime(w_s['time'].values[0])
    YYYY_MM_DD = date.strftime('%Y_%m_%d')
    
    output_file_path = os.path.join(out_path, f"{YYYY_MM_DD}_T00_00_hwe.nc")

    hwe.to_netcdf(path=output_file_path, mode="w")
    print("High wind events have been produced and saved to: ", output_file_path)

# Low wind events


def run_low_wind_events(
    iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path
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
        time + " ENERGY: low wind events computed using the "
        f"energy_onshore application v{version}."
    )

    history = data_u100.attrs["history"] + data_v100.attrs["history"] + message

    # Import processing script.

    u100 = data_u100["u"][:, 0, :, :]
    v100 = data_v100["v"][:, 0, :, :]

    w_s = wind_speed(u100, v100)
   
    lwe = low_wind_events(w_s, threshold=3.0)

    # Global attrs:
    lwe.attrs = {"resolution": data_u100.attrs["resolution"],
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
            "realization": data_u100.attrs["realization"]
            }

    lwe.attrs["history"] = history

    date = pd.to_datetime(w_s['time'].values[0])
    YYYY_MM_DD = date.strftime('%Y_%m_%d')

    output_file_path = os.path.join(out_path, f"{YYYY_MM_DD}_T00_00_lwe.nc")

    lwe.to_netcdf(path=output_file_path, mode="w")
    print("Low wind events have been produced and saved to: ", output_file_path)