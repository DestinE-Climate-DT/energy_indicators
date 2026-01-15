#!/usr/bin/env python3
"""
# Destination Earth: Energy Indicators application
# Authors: Aleksander Lacma-Nadolnik, Francesc Roura-Adserias, Sushovan Ghosh
# Version: 1.1.x

Definition of the runs cripts to be run in the workflow runscript.
"""

# External libraries
import importlib
import os
from datetime import datetime, timezone

import numpy as np
import xarray as xr
import pandas as pd

# Internal libraries
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
    capacity_factor_histogram_opa,
    wind_direction,
)

from energy_onshore.solar import pv_pot



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
    wind_dir.attrs = {"resolution": data_u100.attrs["resolution"],
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

    wind_dir.attrs["history"] = history

    date = pd.to_datetime(wind_dir['time'].values[0])
    YYYY_MM_DD = date.strftime('%Y_%m_%d')

    output_file_path = os.path.join(out_path, f"{YYYY_MM_DD}_T00_00_wd.nc")

    wind_dir.to_netcdf(path=output_file_path, mode="w")
    print("Wind direction has been produced and saved to: ", output_file_path)


# Wind speed anomalies

def run_wind_speed_anomalies(iniyear, inimonth, iniday, in_path, out_path, hpcprojdir, mask=None):
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
        time + f" ENERGY: wind speed anomalies computed using the " 
        "energy_indicators application v{version}."
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

    # Add mask attribute to global attributes if mask was applied
    if mask is not None:
        c_f.attrs["mask"] = "yes"

    c_f.attrs["history"] = history

    date = pd.to_datetime(w_s['time'].values[0])
    YYYY_MM_DD = date.strftime('%Y_%m_%d')

    output_file_path = os.path.join(out_path, f"{YYYY_MM_DD}_T00_00_cf_I.nc")

    c_f.to_netcdf(path=output_file_path, mode="w")
    print("Capacity factor for turbine type 'I' has been produced and saved to: ", output_file_path)


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

    # Add mask attribute to global attributes if mask was applied
    if mask is not None:
        c_f.attrs["mask"] = "yes"

    c_f.attrs["history"] = history

    date = pd.to_datetime(w_s['time'].values[0])
    YYYY_MM_DD = date.strftime('%Y_%m_%d')
    
    output_file_path = os.path.join(out_path, f"{YYYY_MM_DD}_T00_00_cf_II.nc")

    c_f.to_netcdf(path=output_file_path, mode="w")
    print("Capacity factor for turbine type 'II' has been produced and saved to: ", output_file_path)


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

    # Add mask attribute to global attributes if mask was applied
    if mask is not None:
        c_f.attrs["mask"] = "yes"

    c_f.attrs["history"] = history

    date = pd.to_datetime(w_s['time'].values[0])
    YYYY_MM_DD = date.strftime('%Y_%m_%d')
    
    output_file_path = os.path.join(out_path, f"{YYYY_MM_DD}_T00_00_cf_III.nc")

    c_f.to_netcdf(path=output_file_path, mode="w")
    print("Capacity factor for turbine type 'III' has been produced and saved to: ", output_file_path)


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

    # Add mask attribute to global attributes if mask was applied
    if mask is not None:
        c_f.attrs["mask"] = "yes"

    c_f.attrs["history"] = history
    
    date = pd.to_datetime(w_s['time'].values[0])
    YYYY_MM_DD = date.strftime('%Y_%m_%d')
    
    output_file_path = os.path.join(out_path, f"{YYYY_MM_DD}_T00_00_cf_S.nc")

    c_f.to_netcdf(path=output_file_path, mode="w")
    print("Capacity factor for turbine type 'S' has been produced and saved to: ", output_file_path)


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

    # Add mask attribute to global attributes if mask was applied
    if mask is not None:
        cdd.attrs["mask"] = "yes"

    cdd.attrs["history"] = history

    date = pd.to_datetime(data['time'].values[0])
    YYYY_MM_DD = date.strftime('%Y_%m_%d')

    output_file_path = os.path.join(out_path, f"{YYYY_MM_DD}_T00_00_cdd.nc")

    cdd.to_netcdf(path=output_file_path, mode="w")
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

    # Add mask attribute to global attributes if mask was applied
    if mask is not None:
        hdd.attrs["mask"] = "yes"

    hdd.attrs["history"] = history

    date = pd.to_datetime(data['time'].values[0])
    YYYY_MM_DD = date.strftime('%Y_%m_%d')

    output_file_path = os.path.join(out_path, f"{YYYY_MM_DD}_T00_00_hdd.nc")

    hdd.to_netcdf(path=output_file_path, mode="w")
    print("Heating degree days have been produced and saved to: ", output_file_path)


# High wind events

def run_high_wind_events(
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
        f"energy_indicators application v{version}."
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
    
    # Add mask attribute to global attributes if mask was applied
    if mask is not None:
        hwe.attrs["mask"] = "yes"

    hwe.attrs["history"] = history

    date = pd.to_datetime(w_s['time'].values[0])
    YYYY_MM_DD = date.strftime('%Y_%m_%d')
    
    output_file_path = os.path.join(out_path, f"{YYYY_MM_DD}_T00_00_hwe.nc")

    hwe.to_netcdf(path=output_file_path, mode="w")
    print("High wind events have been produced and saved to: ", output_file_path)


# Low wind events

def run_low_wind_events(
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
        f"energy_indicators application v{version}."
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

    # Add mask attribute to global attributes if mask was applied
    if mask is not None:
        lwe.attrs["mask"] = "yes"

    lwe.attrs["history"] = history

    date = pd.to_datetime(w_s['time'].values[0])
    YYYY_MM_DD = date.strftime('%Y_%m_%d')

    output_file_path = os.path.join(out_path, f"{YYYY_MM_DD}_T00_00_lwe.nc")

    lwe.to_netcdf(path=output_file_path, mode="w")
    print("Low wind events have been produced and saved to: ", output_file_path)


# PV potential

def run_pv_potential(
    iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path, mask=None
):
    """
    Compute PV potential (PVP) from hourly temperature (2t), shortwave radiation
    (rsds/avg_sdswrf in W/m2 or ssrd in J/m2), and wind components (u, v).
    Writes a NetCDF and returns None.
    """

    # --- Filenames (hourly raw), mirroring existing pattern ---
    t2_file = (
        f"{iniyear}_{inimonth}_{iniday}_2t_timestep_60_daily_mean.nc"
    )
    avg_sdswrf_file = (
        f"{iniyear}_{inimonth}_{iniday}_avg_sdswrf_timestep_60_daily_mean.nc"
    )
    ws_file = (
        f"{iniyear}_{inimonth}_{iniday}_10si_timestep_60_daily_mean.nc"
    )

    # --- Absolute paths ---
    p_t2   = os.path.join(in_path, t2_file)
    p_avg_sdswrf  = os.path.join(in_path, avg_sdswrf_file)
    p_ws    = os.path.join(in_path, ws_file)

    # --- Open mandatory datasets ---
    data_t2 = xr.open_dataset(p_t2)
    data_G  = xr.open_dataset(p_avg_sdswrf)
    data_ws  = xr.open_dataset(p_ws)


    # --- Metadata (same style as other run_* functions) ---
    time = get_time_utc()
    version = get_application_version()
    message = (
        time + " ENERGY: PV potential computed using the "
        f"energy_indicators application v{version}."
    )

    history = (
        data_t2.attrs["history"]
        + data_G.attrs["history"]
        + data_ws.attrs["history"]
        + message
    )

    # --- Prepare inputs: 2t in °C and wind speed magnitude ---
    t2k = data_t2["2t"]  # Kelvin
    G   = data_G["avg_sdswrf"]
    w_s   = data_ws["10si"]

    t2c = convert_temperature(t2k, unit="C")

    # --- Indicator ---
    from energy_onshore.solar import pv_pot
    PVP = pv_pot(t2c, G, w_s)

    # --- Global attrs (mirror pattern in other run_* functions) ---
    PVP.attrs = {
        "resolution":  t2k.attrs["resolution"],
        "generation":  t2k.attrs["generation"],
        "activity":    t2k.attrs["activity"],
        "dataset":     t2k.attrs["dataset"],
        "stream":      t2k.attrs["stream"],
        "model":       t2k.attrs["model"],
        "experiment":  t2k.attrs["experiment"],
        "levtype":     t2k.attrs["levtype"],
        "expver":      t2k.attrs["expver"],
        "class":       t2k.attrs["class"],
        "type":        t2k.attrs["type"],
        "realization": t2k.attrs["realization"],
    }

    # Add mask attribute to global attributes if mask was applied
    if mask is not None:
        PVP.attrs["mask"] = "yes"

    PVP.attrs["history"] = history

    # --- Output file (consistent naming) ---
    date = pd.to_datetime(PVP["time"].values[0])
    YYYY_MM_DD = date.strftime("%Y_%m_%d")
    output_file_path = os.path.join(out_path, f"{YYYY_MM_DD}_T00_00_pvp.nc")

    PVP.to_netcdf(path=output_file_path, mode="w")
    print("PV potential has been produced and saved to:", output_file_path)


# Capacity factor histogram (OPA)

def run_capacity_factor_histogram_opa(
    iniyear, inimonth, iniday, finyear, finmonth, finday, in_path, out_path, mask=None, cf_type='I'
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
    cf_type : string
        type of capacity factor: I, II, III, S.

    Returns
    -------
    None.

    """
    # Provide the data file name for all variables
    cf_file = (
        f"{iniyear}_{inimonth}_{iniday}_T00_00_cf_{cf_type}.nc"
    )

    absolute_path_cf = os.path.join(in_path, cf_file)

    data_cf = xr.open_dataset(absolute_path_cf)

    cf = data_cf[f"cf_{cf_type.lower()}"]
            
    # Global attrs:
    cf.attrs = data_cf.attrs.copy()
    
    print(f'mask in run_energy_indicators python repo {mask}', flush=True)
    capacity_factor_histogram_opa(cf, in_path, mask=mask, iec_class=f"{cf_type.lower()}")
