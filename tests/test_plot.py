"""
Tests for the ploting functions
"""

import xarray as xr

from energy_indicators.plot import (
    define_colormaps,
    plot_cf,
    plot_wpr,
    plot_degree_days,
    plot_map,
)
from energy_indicators.run_energy_indicators import (
    run_wind_producing_regimes,
    run_cdd,
    run_hdd,
)

# Define colormaps
def test_define_colormaps():
    colormaps = define_colormaps()
    assert isinstance(colormaps, dict)
    for key in ("cmap_cmocean", "cmap_lwe", "cmap_hwe"):
        assert key in colormaps


# Updated plot_map function
def test_plot_map(tmp_path):
    ds = xr.open_dataset("test_data/1990_01_01_T00_00_cf_I.nc")
    da = ds["cf_i"].isel(time=0)
    output_file = tmp_path / "test_map.png"
    plot_map(
        data=da,
        xmin=0,
        xmax=1,
        colormap=define_colormaps()["cmap_cmocean"],
        label="Capacity Factor",
        title="Test Map",
        output_file=str(output_file),
    )
    assert output_file.exists()


# Define the plot_cf function
def test_plot_cf(tmp_path):
    output_dir = tmp_path / "plots"
    plot_cf(
        file_paths={"cf_i": "test_data/1990_01_01_T00_00_cf_I.nc"},
        output_directory=str(output_dir),
        iec_class="I",
    )
    assert any(output_dir.glob("*.png"))


# Define the plot_wpr function
def test_plot_wpr(tmp_path):
    run_wind_producing_regimes("1990", "01", "01", "1990", "01", "01", "test_data/", str(tmp_path))
    wpr_file = next(tmp_path.glob("*_wpr.nc"))

    output_dir = tmp_path / "plots"
    plot_wpr(
        input_file=str(wpr_file),
        output_directory=str(output_dir),
    )
    assert any(output_dir.glob("*.png"))


# define plotting for cdd and hdd days
def test_plot_degree_days(tmp_path):
    run_cdd("1990", "01", "01", "test_data/", str(tmp_path))
    run_hdd("1990", "01", "01", "test_data/", str(tmp_path))
    cdd_file = next(tmp_path.glob("*_cdd.nc"))
    hdd_file = next(tmp_path.glob("*_hdd.nc"))

    output_dir = tmp_path / "plots"
    plot_degree_days(
        input_files={"cdd": str(cdd_file), "hdd": str(hdd_file)},
        output_directory=str(output_dir),
    )
    assert any(output_dir.glob("*.png"))
