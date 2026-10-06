import numpy as np
import pytest
import xarray as xr
import geopandas as gpd
from shapely.geometry import box

from energy_indicators.aggregation import (
    merge_country_boundaries,
    build_region_mask,
    get_or_build_region_mask,
    load_region_mask,
    build_lat_weights,
    load_lat_weights,
    cell_area_km2,
    aggregate_region,
)

"""Tests that correspond to aggregation.py"""


# Two adjacent synthetic countries, boundary at lon=2.3, for testing
# fractional coverage and exact boundary splits.
@pytest.fixture
def two_country_shapefile(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"NUTS_ID": ["AA", "BB"]},
        geometry=[box(0, 40, 2.3, 45), box(2.3, 40, 5, 45)],
        crs="EPSG:4326",
    )
    path = str(tmp_path / "two_country.shp")
    gdf.to_file(path)
    return path


# A country at negative real longitude, for testing 0-360 grid convention
# handling, the shapefile itself is always -180..180 (EPSG:4326), the
# grid passed to build_region_mask is what varies.
@pytest.fixture
def western_hemisphere_shapefile(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"NUTS_ID": ["US"]},
        geometry=[box(-115, 30, -95, 45)],
        crs="EPSG:4326",
    )
    path = str(tmp_path / "western.shp")
    gdf.to_file(path)
    return path


@pytest.fixture
def small_grid():
    lat = np.linspace(40.5, 44.5, 10)
    lon = np.linspace(0.25, 4.75, 10)
    return lat, lon


@pytest.fixture
def small_region_mask(two_country_shapefile, small_grid):
    lat, lon = small_grid
    return build_region_mask(lat, lon, two_country_shapefile, id_col="NUTS_ID")


# build_region_mask


def test_build_region_mask_coverage_sums_to_one(small_region_mask):
    # interior cells only, this grid deliberately extends slightly past
    # the two synthetic countries at its outer edge, where coverage is
    # legitimately less than 1
    interior = small_region_mask.sum(dim="region").isel(lat=slice(1, -1), lon=slice(1, -1))
    assert np.allclose(interior.values, 1.0, atol=1e-5)


def test_build_region_mask_boundary_split_is_exact(small_region_mask, small_grid):
    lat, lon = small_grid
    # cell nearest lon=2.25 spans [2.0, 2.5], border at 2.3 -> 60% AA, 40% BB
    idx = int(np.argmin(np.abs(lon - 2.25)))
    aa_frac = float(small_region_mask.sel(region="AA").isel(lon=idx).values[0])
    bb_frac = float(small_region_mask.sel(region="BB").isel(lon=idx).values[0])
    assert np.isclose(aa_frac, 0.6, atol=1e-6)
    assert np.isclose(bb_frac, 0.4, atol=1e-6)


def test_build_region_mask_sequential_matches_parallel(two_country_shapefile, small_grid):
    lat, lon = small_grid
    seq = build_region_mask(lat, lon, two_country_shapefile, id_col="NUTS_ID", n_jobs=1)
    par = build_region_mask(lat, lon, two_country_shapefile, id_col="NUTS_ID", n_jobs=2)
    assert np.array_equal(seq.values, par.values)


def test_build_region_mask_rejects_irregular_grid(two_country_shapefile):
    lat = np.array([40.0, 41.0, 43.0])
    lon = np.linspace(0, 5, 10)
    with pytest.raises(ValueError):
        build_region_mask(lat, lon, two_country_shapefile, id_col="NUTS_ID")


def test_build_region_mask_handles_0_360_longitude(western_hemisphere_shapefile):
    # regression: an earlier version placed grid cells at their literal
    # lon value, so a cell at lon=250 (0..360 convention, really -110)
    # was compared against shapefiles defined in -180..180, silently
    # giving zero coverage across the whole western hemisphere
    lat = np.arange(25, 50, 0.5)
    lon = np.arange(230, 300, 0.5)
    mask = build_region_mask(lat, lon, western_hemisphere_shapefile, id_col="NUTS_ID")
    assert float(mask.sum()) > 100  # was exactly 0 before the fix


# get_or_build_region_mask


def test_get_or_build_region_mask_caches(two_country_shapefile, small_grid, tmp_path):
    lat, lon = small_grid
    cache_dir = str(tmp_path / "cache")
    first = get_or_build_region_mask(lat, lon, two_country_shapefile, id_col="NUTS_ID", cache_dir=cache_dir)
    second = get_or_build_region_mask(lat, lon, two_country_shapefile, id_col="NUTS_ID", cache_dir=cache_dir)
    assert np.array_equal(first.values, second.values)


def test_get_or_build_region_mask_different_id_col_not_stale(two_country_shapefile, small_grid, tmp_path):
    # regression: the cache key used to depend only on grid and shapefile,
    # not id_col, so a second call with a different id_col against the
    # same grid/shapefile silently returned the first call's mask
    lat, lon = small_grid
    gdf = gpd.read_file(two_country_shapefile)
    gdf["alt_id"] = ["region_aa", "region_bb"]
    gdf.to_file(two_country_shapefile)

    cache_dir = str(tmp_path / "cache")
    first = get_or_build_region_mask(lat, lon, two_country_shapefile, id_col="NUTS_ID", cache_dir=cache_dir)
    second = get_or_build_region_mask(lat, lon, two_country_shapefile, id_col="alt_id", cache_dir=cache_dir)
    assert list(first.region.values) != list(second.region.values)


# load_region_mask / load_lat_weights


def test_load_region_mask_renames_latitude_longitude(small_region_mask, tmp_path):
    renamed = small_region_mask.rename({"lat": "latitude", "lon": "longitude"})
    path = str(tmp_path / "mask.nc")
    renamed.to_dataset(name="mask").to_netcdf(path)
    loaded = load_region_mask(path)
    assert "lat" in loaded.dims and "lon" in loaded.dims


def test_load_lat_weights_renames_latitude_longitude(small_grid, tmp_path):
    lat, lon = small_grid
    weights = build_lat_weights(lat, lon).rename({"lat": "latitude", "lon": "longitude"})
    path = str(tmp_path / "latw.nc")
    weights.to_dataset(name="lat_weights").to_netcdf(path)
    loaded = load_lat_weights(path)
    assert "lat" in loaded.dims and "lon" in loaded.dims


# build_lat_weights / cell_area_km2


def test_build_lat_weights_matches_cosine(small_grid):
    lat, lon = small_grid
    weights = build_lat_weights(lat, lon)
    expected = np.cos(np.deg2rad(lat))
    assert np.allclose(weights.isel(lon=0).values, expected, atol=1e-5)


def test_cell_area_km2_matches_known_country_area(small_region_mask, small_grid):
    lat, lon = small_grid
    data = xr.DataArray(np.ones((len(lat), len(lon))), dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    total = aggregate_region(data, small_region_mask, stat="sum")
    # AA+BB together span roughly 5deg lon x 5deg lat around 42N, order
    # of magnitude check only, synthetic shapefile.
    combined = float(total.sum())
    assert 100_000 < combined < 400_000


# aggregate_region


def test_aggregate_region_mean_of_constant_field_is_one(small_region_mask, small_grid):
    lat, lon = small_grid
    lat_weights = build_lat_weights(lat, lon)
    data = xr.DataArray(np.ones((len(lat), len(lon))), dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    out = aggregate_region(data, small_region_mask, lat_weights, stat="mean")
    assert np.allclose(out.values, 1.0)


def test_aggregate_region_sum_boundary_cell_weighted_by_fraction(small_region_mask, small_grid):
    # regression: sum used to include any covered cell at full weight
    # regardless of fractional coverage
    lat, lon = small_grid
    data = xr.DataArray(np.ones((len(lat), len(lon))), dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    out = aggregate_region(data, small_region_mask, stat="sum")
    area = cell_area_km2(lat, lon)
    expected = (small_region_mask * area).sum(dim=("lat", "lon"))
    assert np.allclose(out.values, expected.values)


def test_aggregate_region_dtype_upcast_avoided(small_region_mask, small_grid):
    # regression: multiplying float64 data by a float32 mask silently
    # upcast every intermediate to float64, doubling memory for no
    # precision benefit
    lat, lon = small_grid
    data64 = xr.DataArray(
        np.random.uniform(3, 15, (len(lat), len(lon))).astype("float64"),
        dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
    )
    out = aggregate_region(data64, small_region_mask, stat="mean")
    assert out.dtype == small_region_mask.dtype


def test_aggregate_region_accepts_1d_lat_weights(small_region_mask, small_grid):
    # regression: lat_weights was assumed to always have both lat and lon
    # dims, but some real sources (e.g. CDS's global LATW file) are 1D,
    # lat only
    lat, lon = small_grid
    lat_weights_1d = xr.DataArray(np.cos(np.deg2rad(lat)).astype("float32"), dims=("lat",), coords={"lat": lat})
    data = xr.DataArray(np.ones((len(lat), len(lon))), dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    out = aggregate_region(data, small_region_mask, lat_weights_1d, stat="mean")
    assert np.allclose(out.values, 1.0)


@pytest.mark.parametrize("stat", ["mean", "sum", "min", "max", "std"])
def test_aggregate_region_sequential_matches_parallel(small_region_mask, small_grid, stat):
    lat, lon = small_grid
    lat_weights = build_lat_weights(lat, lon)
    data = xr.DataArray(
        np.random.uniform(3, 15, (len(lat), len(lon))).astype("float64"),
        dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
    )
    seq = aggregate_region(data, small_region_mask, lat_weights, stat=stat, n_jobs=1)
    par = aggregate_region(data, small_region_mask, lat_weights, stat=stat, n_jobs=2)
    assert np.array_equal(seq.values, par.values, equal_nan=True)


def test_aggregate_region_rejects_unsupported_stat(small_region_mask, small_grid):
    lat, lon = small_grid
    data = xr.DataArray(np.ones((len(lat), len(lon))), dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    with pytest.raises(AssertionError):
        aggregate_region(data, small_region_mask, stat="median")


# merge_country_boundaries


def test_merge_country_boundaries_dedups_and_prefers_fine_source(tmp_path):
    fine = gpd.GeoDataFrame(
        {"NUTS_ID": ["FR"], "ISO3_CODE": ["FRA"], "NAME_ENGL": ["France"]},
        geometry=[box(-5, 41, 8, 51)],
        crs="EPSG:4326",
    )
    fine_path = str(tmp_path / "fine.shp")
    fine.to_file(fine_path)

    coarse = gpd.GeoDataFrame(
        {"ADM0_A3": ["FRA", "USA"], "NAME_EN": ["France (coarse)", "United States"]},
        geometry=[box(-6, 40, 9, 52), box(-125, 25, -65, 49)],
        crs="EPSG:4326",
    )
    coarse_path = str(tmp_path / "coarse.shp")
    coarse.to_file(coarse_path)

    merged = merge_country_boundaries(
        fine_path, coarse_path,
        nuts_id_col="NUTS_ID", nuts_iso3_col="ISO3_CODE", nuts_name_col="NAME_ENGL",
        admin_iso3_col="ADM0_A3", admin_name_col="NAME_EN",
    )

    assert len(merged) == 2
    fr_row = merged[merged.region == "FRA"].iloc[0]
    assert fr_row.source == "nuts"
    usa_row = merged[merged.region == "USA"].iloc[0]
    assert usa_row.source == "admin"

def test_aggregate_region_tolerates_coordinate_float_noise():
    lat = np.array([0.0, 0.05, 0.10, 0.15])
    lon = np.array([0.0, 0.05, 0.10, 0.15])
    data = xr.DataArray(
        np.full((1, 4, 4), 2.0),
        dims=("time", "lat", "lon"),
        coords={"time": [0], "lat": lat, "lon": lon},
        name="x",
    )
    mask = xr.DataArray(
        np.ones((1, 4, 4), dtype=np.float32),
        dims=("region", "lat", "lon"),
        coords={"region": ["AAA"], "lat": lat + 4e-11, "lon": lon + 4e-11},
    )
    out = aggregate_region(data, mask, stat="mean")
    assert float(out.isel(time=0, region=0)) == pytest.approx(2.0)

def test_run_nuts_aggregation_defaults_to_cosine_lat_weights(tmp_path):
    from energy_indicators.run_energy_indicators import run_nuts_aggregation

    lat = np.array([0.0, 60.0])
    lon = np.array([0.0, 10.0])
    time = np.array(["2010-08-01"], dtype="datetime64[ns]")

    field = np.array([[[10.0, 10.0], [20.0, 20.0]]], dtype="float32")  # (time, lat, lon)
    xr.Dataset(
        {"cdd": (("time", "lat", "lon"), field)},
        coords={"time": time, "lat": lat, "lon": lon},
    ).to_netcdf(tmp_path / "2010_08_01_T00_00_cdd.nc")

    mask = np.ones((1, 2, 2), dtype="float32")
    xr.Dataset(
        {"mask": (("region", "lat", "lon"), mask)},
        coords={"region": ["AAA"], "lat": lat, "lon": lon},
    ).to_netcdf(tmp_path / "mask.nc")

    run_nuts_aggregation(
        "2010", "08", "01", "cdd", str(tmp_path), str(tmp_path), str(tmp_path / "mask.nc"),
    )
    out = xr.open_dataset(tmp_path / "2010_08_01_T00_00_cdd_nuts0.nc")["cdd"]
    # weights cos(0)=1 and cos(60)=0.5: (2*10*1 + 2*20*0.5) / 3 = 13.33, not the unweighted 15
    assert float(out.isel(time=0, region=0)) == pytest.approx(40 / 3, rel=1e-4)

