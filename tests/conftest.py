import pytest
import pandas as pd

import numpy as np

from energy_onshore.core import create_dataset


def create_template_arrays():
    """
    Creates a template array to be fed into the fixtures.

    Returns
    -------
    t_c : numpy array
        temperature array, in Celsius.
    t_k : numpy array
        temperature array, in Kelvin.
    u : numpy array
        zonal component of windspeed.
    v : numpy array
        meridional component of windspeed.
    attributes : dict
        attribute metadata that is added by default in this test array.
    coordinates : dict
        lat, lon, time.
    dimensions : list
        names of the dimentions (lat lon time).

    """
    # Define the sizes of the dimensions
    lat_size = 10
    lon_size = 10
    time_size = 7

    # Create coordinates
    latitudes = np.linspace(-90, 90, lat_size)
    longitudes = np.linspace(-180, 180, lon_size)
    times = pd.date_range("2023-01-01", periods=time_size)

    # Create data filled with values close to 20
    t_c = 20 + np.random.randn(time_size, lat_size, lon_size)
    t_k = 273.15 + 20 + np.random.randn(time_size, lat_size, lon_size)
    u = np.random.uniform(0, 30, (time_size, lat_size, lon_size))
    v = np.random.uniform(0, 30, (time_size, lat_size, lon_size))

    # Define variables, coordinates, dimensions, and attributes
    t_c = {"t_c": t_c}

    t_k = {"t_k": t_k}

    u = {"u": u}

    v = {"v": v}

    coordinates = {"time": times, "lat": latitudes, "lon": longitudes}

    dimensions = ( "time", "lat", "lon")

    attributes = {
        "units": "depending on de variab",
    }
    return t_c, t_k, u, v, attributes, coordinates, dimensions


t_c, t_k, u, v, attributes, coordinates, dimensions = create_template_arrays()


@pytest.fixture
def dataarray_t_c():
    """
    Creates a dataarray filled with random temperature values in Celsius.

    Returns
    -------
    xr.DataArray
        temperature celsius data array

    """
    test_dataset = create_dataset(t_c, {"units": "C"}, coordinates, dimensions)
    return test_dataset.to_array()


@pytest.fixture
def dataarray_t_k():
    """
    Creates a dataarray filled with random temperature values in Kelvin.    

    Returns
    -------
    xr.DataArray
        temperature kelvin data array

    """
    test_dataset = create_dataset(t_k, {"units": "K"}, coordinates, dimensions)
    return test_dataset.to_array()


@pytest.fixture
def dataarray_u():
    """
    Creates a dataarray filled with random zonal wind values in ms⁻¹.

    Returns
    -------
    xr.DataArray
        u data array

    """
    test_dataset = create_dataset(u, {"units": "ms-1"}, coordinates, dimensions)
    return test_dataset.to_array()


@pytest.fixture
def dataarray_v():
    """
    Creates a dataarray filled with random meridional wind values in ms⁻¹.    

    Returns
    -------
    xr.DataArray
        v data array

    """
    test_dataset = create_dataset(v, {"units": "ms-1"}, coordinates, dimensions)
    return test_dataset.to_array()
