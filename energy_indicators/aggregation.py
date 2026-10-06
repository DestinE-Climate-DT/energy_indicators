#!/usr/bin/env python3
"""
| Destination Earth: Energy Indicators application (spatial aggregation)

Reduces gridded (time, lat, lon) indicator fields to one value per region
(e.g. country) using a precomputed fractional-coverage region mask.

Workflow, in order:

1. merge_country_boundaries : combine two region-polygon sources into
   one global set (call once, cache the result to disk).
2. build_region_mask : rasterise those polygons onto a specific grid,
   as a fractional-coverage array (call once per grid, expensive).
3. get_or_build_region_mask : wraps (2) with a cache keyed to the exact
   grid and parameters used, so step 2 only runs the first time a given
   grid/config is seen.
4. build_lat_weights / load_lat_weights : latitude-area weighting for
   regular lat-lon grids.
5. aggregate_region : apply a region mask (from step 2/3) to an
   indicator field. Called once per indicator file, indicator-agnostic
   - does not know or care whether the input is CDD, WPD, CF, etc.
   Cheap relative to step 2, but at native grid resolution the mask
   itself can be tens of GB, so this is memory-bandwidth bound rather
   than instant, use n_jobs when calling this at that scale.
"""

# External libraries
import xarray as xr
import numpy as np
import pandas as pd
import warnings
import os
import hashlib

# Internal libraries
from .core import get_type

SUPPORTED_STATS = {"mean", "sum", "min", "max", "std"}


def merge_country_boundaries(
    nuts_shapefile,
    admin_shapefile,
    nuts_id_col="NUTS_ID",
    nuts_iso3_col="ISO3_CODE",
    nuts_name_col="NAME_ENGL",
    admin_iso3_col="ADM0_A3",
    admin_name_col="NAME_EN",
    output_shapefile=None,
):
    """
    Merge two sets of country-level region polygons into one global set:
    a fine-detail regional source (e.g. Eurostat NUTS0) and a coarser but
    globally-complete source (e.g. Natural Earth ADMIN0). For any country
    present in the fine-detail source, its polygon is kept and the
    corresponding coarse-source polygon is dropped, so each country
    appears exactly once, from whichever source is more detailed for it.

    Matching between the two sources is done on ISO3 codes rather than
    2-letter codes, since 2-letter national codes and ISO 3166-1 alpha-2
    disagree for a few countries (e.g. Greece, United Kingdom), while
    ISO3 codes are unambiguous on both sides. Output region IDs are ISO3
    throughout, for a single consistent ID scheme across the whole set.

    This is a one-off preprocessing step: run once, save the result, and
    pass that single file to build_region_mask/get_or_build_region_mask
    like any other shapefile from then on.

    Input
    -------
    nuts_shapefile: str
        Path to the fine-detail regional shapefile.
    admin_shapefile: str
        Path to the coarse, globally-complete shapefile.
    nuts_id_col, nuts_iso3_col, nuts_name_col: str
        Column names in nuts_shapefile for its native region ID (kept as
        an auxiliary "nuts_id" coordinate only), the ISO3 code (used both
        for matching against admin_shapefile and as the output "region"
        ID), and a human-readable name.
    admin_iso3_col, admin_name_col: str
        Column names in admin_shapefile for its ISO3 identifier and name.
        Verify these against the actual file before relying on them -
        different distributions/versions of global boundary datasets use
        different column names, and some ISO3-like fields contain
        placeholder values for a handful of disputed/dependent
        territories rather than a real code.
    output_shapefile: str or None
        If given, the merged GeoDataFrame is also written to this path.
        Use a format without a field-name-length limit (e.g. GeoPackage,
        ".gpkg") - formats that truncate long field names will silently
        break the "region_name" column without raising an error.

    References
    -------
    Fine-detail regional source (nuts_shapefile): Eurostat GISCO NUTS,
        https://gisco-services.ec.europa.eu/distribution/v2/nuts/
        (this function was built/tested against NUTS_RG_01M_2024_4326_LEVL_0,
        1:1M scale, EPSG:4326).
    Coarse global source (admin_shapefile): Natural Earth admin_0_countries,
        https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/
        (built/tested against the 1:10m version).

    Output
    -------
    merged: geopandas.GeoDataFrame
        Columns: "region" (ISO3 code, consistent across both sources),
        "nuts_id" (native fine-detail-source code where applicable, else
        empty), "region_name", "geometry", "source" ("nuts" or "admin",
        for traceability).
    """
    import geopandas as gpd

    nuts = gpd.read_file(nuts_shapefile).to_crs("EPSG:4326")
    admin = gpd.read_file(admin_shapefile).to_crs("EPSG:4326")

    for col, name, df in [
        (nuts_id_col, "nuts_id_col", nuts),
        (nuts_iso3_col, "nuts_iso3_col", nuts),
        (admin_iso3_col, "admin_iso3_col", admin),
    ]:
        assert col in df.columns, (
            f'"{name}"={col!r} not found in its shapefile columns: {list(df.columns)}.'
        )

    nuts_iso3_codes = set(nuts[nuts_iso3_col].dropna().unique())

    nuts_out = nuts[[nuts_iso3_col, nuts_id_col, nuts_name_col, "geometry"]].rename(
        columns={nuts_iso3_col: "region", nuts_id_col: "nuts_id", nuts_name_col: "region_name"}
    )
    nuts_out["source"] = "nuts"

    admin_rest = admin[~admin[admin_iso3_col].isin(nuts_iso3_codes)]
    admin_out = admin_rest[[admin_iso3_col, admin_name_col, "geometry"]].rename(
        columns={admin_iso3_col: "region", admin_name_col: "region_name"}
    )
    admin_out["nuts_id"] = ""
    admin_out["source"] = "admin"

    merged = pd.concat([nuts_out, admin_out], ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")

    n_dupes = merged["region"].duplicated().sum()
    if n_dupes:
        warnings.warn(
            f"{n_dupes} duplicate region IDs after merging - check for ID "
            "collisions between the two sources.",
            UserWarning,
        )

    if output_shapefile is not None:
        if output_shapefile.lower().endswith((".shp", ".shp.zip")):
            warnings.warn(
                'Saving to ESRI Shapefile (.shp) truncates field names to 10 '
                'characters - "region_name" silently becomes "region_nam" with '
                "no error, which then breaks name_col lookups downstream. "
                'Use a ".gpkg" (GeoPackage) path instead.',
                UserWarning,
            )
        merged.to_file(output_shapefile)

    return merged


def _wrap_longitude(lon):
    """
    Wrap longitude values to the standard -180..180 convention used by
    EPSG:4326 shapefiles. Grid data is commonly published on a 0..360
    convention instead (e.g. lon running 0 to 359.95) - using those values
    directly to build cell geometry would place every cell with lon > 180
    at the wrong position entirely (e.g. lon=250 taken literally, instead
    of the -110 it actually represents), silently giving zero coverage
    for the whole affected hemisphere.
    """
    return ((np.asarray(lon) + 180) % 360) - 180


def _region_overlay(args):
    """
    Worker: compute fractional coverage for one region (or one tile of one
    region - see build_region_mask's tiling) against a bounding-box-
    restricted subset of the grid. Module-level (not a closure) so it can
    be pickled for ProcessPoolExecutor.

    lat/lon passed in are always the grid's FULL original coordinate
    arrays (never a sliced-down subset) - this is what guarantees indices
    computed here are always correct global positions, with or without
    tiling. idx_range (lat_lo, lat_hi, lon_lo, lon_hi), if given, narrows
    the search to that index sub-range on top of the region's own bbox -
    purely a search-space restriction for splitting one region's work
    across multiple tasks, not a change to what gets computed.
    """
    region_id, region_geom, lat, lon, dlat, dlon, idx_range = args
    from shapely.geometry import box
    import geopandas as gpd

    lon_wrapped = _wrap_longitude(lon)

    minx, miny, maxx, maxy = region_geom.bounds
    lat_mask = (lat >= miny - dlat) & (lat <= maxy + dlat)
    lon_mask = (lon_wrapped >= minx - dlon) & (lon_wrapped <= maxx + dlon)

    if idx_range is not None:
        lat_lo, lat_hi, lon_lo, lon_hi = idx_range
        tile_lat_mask = np.zeros(len(lat), dtype=bool)
        tile_lat_mask[lat_lo:lat_hi] = True
        tile_lon_mask = np.zeros(len(lon), dtype=bool)
        tile_lon_mask[lon_lo:lon_hi] = True
        lat_mask &= tile_lat_mask
        lon_mask &= tile_lon_mask

    sub_lat_idx = np.nonzero(lat_mask)[0]
    sub_lon_idx = np.nonzero(lon_mask)[0]
    empty = (
        region_id,
        np.empty((0,), dtype=np.int64),
        np.empty((0,), dtype=np.int64),
        np.empty((0,), dtype="float32"),
    )
    if len(sub_lat_idx) == 0 or len(sub_lon_idx) == 0:
        return empty

    sub_lat = lat[sub_lat_idx]
    sub_lon_wrapped = lon_wrapped[sub_lon_idx]
    lat_idx_grid, lon_idx_grid = np.meshgrid(sub_lat_idx, sub_lon_idx, indexing="ij")
    lat_val_grid, lon_val_grid = np.meshgrid(sub_lat, sub_lon_wrapped, indexing="ij")
    cells = [
        box(lo - dlon / 2, la - dlat / 2, lo + dlon / 2, la + dlat / 2)
        for la, lo in zip(lat_val_grid.ravel(), lon_val_grid.ravel())
    ]
    cells_gdf = gpd.GeoDataFrame(
        {"lat_idx": lat_idx_grid.ravel(), "lon_idx": lon_idx_grid.ravel()},
        geometry=cells,
        crs="EPSG:4326",
    )
    region_gdf = gpd.GeoDataFrame({"region": [region_id]}, geometry=[region_geom], crs="EPSG:4326")

    with warnings.catch_warnings():
        # .area is inaccurate in an unprojected (lat/lon) CRS for absolute
        # areas, but harmless here: only used as a ratio of two areas
        # computed the same way within the same cell, so the latitude-
        # dependent distortion cancels out.
        warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")
        overlap = gpd.overlay(cells_gdf, region_gdf, how="intersection")
        if overlap.empty:
            return empty
        frac = (overlap.geometry.area / (dlat * dlon)).values.astype("float32")

    return region_id, overlap["lat_idx"].values, overlap["lon_idx"].values, frac


def build_region_mask(
    lat, lon, shapefile, id_col, name_col=None, simplify_tolerance=None,
    n_jobs=1, max_tiles_per_region=16,
):
    """
    Generate a fractional-coverage region mask at an arbitrary regular
    lat-lon grid, via exact polygon-grid-cell intersection: for every
    (region, grid cell) pair, what fraction of that cell's area falls
    inside that region's polygon.

    Expensive - run once per (grid, region-set, id_col) combination and
    reuse the result (see get_or_build_region_mask for automatic caching).
    Do not call this inside a per-indicator-file loop; aggregate_region is
    the cheap, repeatable counterpart for that.

    Processes one region at a time, each restricted first to its own
    bounding-box subset of the grid, then overlaid independently. This
    doesn't reduce the total computation on a single core (the exact-
    intersection computation itself dominates regardless of how the work
    is split), but it does make each region's work independent, so it
    parallelises across n_jobs worker processes.

    Input
    -------
    lat: array-like
        1D array of latitude values of the target (regular) grid.
    lon: array-like
        1D array of longitude values of the target (regular) grid.
    shapefile: str
        Path to a region-polygon shapefile readable by geopandas (e.g.
        merge_country_boundaries's output). Zipped shapefiles (.shp.zip)
        can be passed directly, no need to unzip.
    id_col: str
        Column in the shapefile holding the region identifier.
    name_col: str or None
        Optional column holding a human-readable region name, stored as
        an auxiliary "region_name" coordinate alongside "region".
    simplify_tolerance: float or None
        Angular simplification tolerance (degrees) applied to the region
        polygons before the overlay. Overlay runtime is dominated by
        polygon vertex count far more than by cell count, so a high-detail
        source shapefile (e.g. carrying full coastline detail) costs far
        more than a several-km grid can actually resolve. None defaults
        to 0.4x the grid spacing; pass 0 to disable simplification
        entirely.
    n_jobs: int
        Number of worker processes for parallel per-region overlay.
        Default 1 (sequential). Set to the number of available cores when
        running on a multi-core node for large domains/region sets.
    max_tiles_per_region: int
        Cap on how many tiles a single region can be split into. Tile
        count scales automatically with each region's own grid-cell
        footprint relative to the median region's size, so this cap only
        binds for the very largest few regions (e.g. Russia, Canada) -
        most regions, already at or below typical size, aren't split at
        all. Only takes effect with n_jobs > 1. Tiles are disjoint,
        contiguous index sub-ranges of the region's own bounding box, so
        this only changes how work is scheduled across cores, never what
        gets computed. Default 16.

    Output
    -------
    region_mask: xarray.DataArray ; (region, lat, lon)
        Fractional coverage (0-1) of each region in each grid cell.

    Notes
    -------
    Assumes a regular grid (uniform spacing in both lat and lon) - raises
    if that assumption doesn't hold. Fractional coverage is computed as a
    ratio of two areas in EPSG:4326 degrees^2 (intersection area / cell
    area); this is valid despite latitude-dependent physical-area
    distortion, because both areas are distorted identically within a
    single cell, so the ratio is unaffected. Latitude weighting for
    physical area (cosine-latitude) is applied separately, at
    aggregate_region time via lat_weights - kept out of this function so
    region_mask stays a pure coverage fraction.

    Longitude convention: works with either -180..180 or 0..360 input
    (longitude is wrapped internally for geometry construction, but the
    output region_mask keeps the original lon values, so it still aligns
    against data expressed in either convention). Not currently handled:
    grid cells whose box would straddle the antimeridian (lon +-180) get
    an unwrapped, and therefore likely undercounted, geometry - affects at
    most a handful of cells directly on the date line, mostly open ocean.
    """
    import geopandas as gpd

    gdf = gpd.read_file(shapefile)
    if gdf.empty:
        raise ValueError(f"Empty shapefile: {shapefile}")
    if gdf.crs is None:
        raise ValueError(f"Missing CRS in shapefile: {shapefile}")
    gdf = gdf.to_crs("EPSG:4326")

    lat = np.asarray(lat)
    lon = np.asarray(lon)
    dlat = abs(lat[1] - lat[0])
    dlon = abs(lon[1] - lon[0])
    if not (np.allclose(np.diff(lat), dlat) and np.allclose(np.diff(lon), dlon)):
        raise ValueError(
            "build_region_mask assumes a regular grid (uniform lat/lon spacing); "
            "got non-uniform spacing."
        )

    if simplify_tolerance is None:
        simplify_tolerance = 0.4 * min(dlat, dlon)
    if simplify_tolerance > 0:
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.simplify(simplify_tolerance)

    region_ids = gdf[id_col].values
    geometries = gdf.geometry.values
    lon_wrapped = _wrap_longitude(lon)

    def _candidate_cells(geom):
        """Grid cells within this region's bbox - a more direct cost proxy
        than raw bbox area in degrees, since it accounts for actual grid
        resolution and doesn't get skewed by a region's degree-space shape."""
        minx, miny, maxx, maxy = geom.bounds
        cand_lat_idx = np.nonzero((lat >= miny - dlat) & (lat <= maxy + dlat))[0]
        cand_lon_idx = np.nonzero((lon_wrapped >= minx - dlon) & (lon_wrapped <= maxx + dlon))[0]
        return cand_lat_idx, cand_lon_idx

    cell_counts = np.array([
        len(c[0]) * len(c[1]) for c in (_candidate_cells(g) for g in geometries)
    ])

    # Submit the largest/most expensive regions first (by grid-cell count),
    # so they start immediately in parallel with everything else rather
    # than only being reached once many small regions have already
    # finished and most workers have gone idle. Affects scheduling order
    # only - results are scattered back into the final array by region ID
    # regardless of completion order.
    task_order = np.argsort(cell_counts)[::-1]

    # Split each region into tiles proportional to its own size relative to
    # a "typical" region (the median cell count across all regions), so
    # giant countries (Russia, Canada, ...) get split enough to occupy
    # several cores at once and finish in roughly the time of a mid-sized
    # country, while most regions - already at or below typical size -
    # aren't split at all. Capped at max_tiles_per_region to avoid
    # fragmenting the very largest into more pieces than useful. Purely a
    # scheduling change: tiles are disjoint, contiguous index sub-ranges,
    # so every cell is computed exactly once regardless of tiling - same
    # result as untiled. Only applied with n_jobs > 1 (sequential execution
    # gains nothing from more, smaller tasks, only overhead).
    # Reference "large" size for deciding how many tiles a region needs.
    # Most of the 259-ish regions in a typical global set are small
    # countries/territories, so the median cell count is dominated by
    # them and would end up near-zero - using it as the split threshold
    # would split nearly everything, including tiny regions where tiling
    # overhead outweighs any benefit. A high percentile, plus an absolute
    # floor below which nothing gets split regardless of relative size,
    # keeps tiling limited to genuine outliers.
    reference_cells = max(np.percentile(cell_counts, 90), 1)
    min_cells_to_split = 2000

    tasks = []
    tile_counts_log = []
    for i in task_order:
        region_id = region_ids[i]
        geom = geometries[i]
        cand_lat_idx, cand_lon_idx = _candidate_cells(geom)
        if len(cand_lat_idx) == 0 or len(cand_lon_idx) == 0:
            tasks.append((region_id, geom, lat, lon, dlat, dlon, None))
            continue

        n_tiles_target = 1
        if n_jobs > 1 and cell_counts[i] >= min_cells_to_split:
            n_tiles_target = int(np.clip(
                np.ceil(cell_counts[i] / reference_cells), 1, max_tiles_per_region,
            ))

        if n_tiles_target == 1:
            tasks.append((region_id, geom, lat, lon, dlat, dlon, None))
            continue

        # Factor into a lat x lon tile grid roughly matching this region's
        # own aspect ratio, so tiles stay reasonably square rather than
        # long thin strips.
        aspect = len(cand_lat_idx) / len(cand_lon_idx)
        n_lat_tiles = max(1, round((n_tiles_target * aspect) ** 0.5))
        n_lat_tiles = min(n_lat_tiles, len(cand_lat_idx))
        n_lon_tiles = max(1, int(np.ceil(n_tiles_target / n_lat_tiles)))
        n_lon_tiles = min(n_lon_tiles, len(cand_lon_idx))
        tile_counts_log.append((str(region_id), int(cell_counts[i]), n_lat_tiles * n_lon_tiles))

        lat_blocks = np.array_split(np.arange(cand_lat_idx.min(), cand_lat_idx.max() + 1), n_lat_tiles)
        lon_blocks = np.array_split(np.arange(cand_lon_idx.min(), cand_lon_idx.max() + 1), n_lon_tiles)
        for lat_block in lat_blocks:
            for lon_block in lon_blocks:
                if len(lat_block) == 0 or len(lon_block) == 0:
                    continue
                idx_range = (
                    int(lat_block.min()), int(lat_block.max()) + 1,
                    int(lon_block.min()), int(lon_block.max()) + 1,
                )
                tasks.append((region_id, geom, lat, lon, dlat, dlon, idx_range))

    if tile_counts_log:
        print(f"Reference size (90th percentile): {reference_cells:.0f} cells, min size to split: {min_cells_to_split}. Split regions (region, cells, n_tiles):")
        for region_id, cells, n_tiles in tile_counts_log:
            print(f"  {region_id}: {cells:,} cells -> {n_tiles} tiles")

    n_tasks = len(tasks)
    if n_jobs == 1:
        results = []
        for i, task in enumerate(tasks):
            results.append(_region_overlay(task))
            print(f"  [{i + 1}/{n_tasks}] {task[0]} done", flush=True)
    else:
        import concurrent.futures

        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_region_overlay, task): task[0] for task in tasks}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                region_id = futures[future]
                results.append(future.result())
                print(f"  [{i + 1}/{n_tasks}] {region_id} done", flush=True)

    region_index = {region_id: i for i, region_id in enumerate(region_ids)}
    frac = np.zeros((len(region_ids), len(lat), len(lon)), dtype="float32")
    for region_id, lat_idx, lon_idx, frac_vals in results:
        if len(frac_vals) == 0:
            continue
        frac[region_index[region_id], lat_idx, lon_idx] = frac_vals

    region_mask = xr.DataArray(
        frac,
        dims=("region", "lat", "lon"),
        coords={"region": region_ids, "lat": lat, "lon": lon},
        name="mask",
    )
    if name_col is not None and name_col in gdf.columns:
        region_mask = region_mask.assign_coords(region_name=("region", gdf[name_col].values))

    return region_mask


def _rename_to_lat_lon(da):
    """
    Rename latitude/longitude dims to lat/lon if present, to match this
    package's (time, lat, lon) convention used throughout wind.py/demand.py.
    """
    rename = {}
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    return da.rename(rename) if rename else da


def load_region_mask(mask_file, region_dim="region"):
    """
    Load a precomputed fractional-coverage region mask and align it to
    this package's (region, lat, lon) convention. Does NOT generate the
    mask - only loads one that already exists (see build_region_mask for
    generation).

    Input
    -------
    mask_file: str
        Path to a NetCDF file with a mask variable of dims (region, lat, lon)
        or (region, latitude, longitude), where each region-slice holds the
        fractional coverage of that region in each grid cell (0-1).
    region_dim: str
        Name of the region dimension/coordinate in the file. Default "region".

    Output
    -------
    region_mask: xarray.DataArray ; (region, lat, lon)
        Fractional coverage mask, coordinate names normalised to lat/lon.
    """
    mask_ds = xr.open_dataset(mask_file)
    var_name = list(mask_ds.data_vars)[0]
    region_mask = mask_ds[var_name]

    assert region_dim in region_mask.dims, (
        f'Expected a "{region_dim}" dimension in {mask_file}, found {region_mask.dims}.'
    )

    return _rename_to_lat_lon(region_mask)


def _grid_signature(lat, lon):
    """
    Deterministic short hash of a grid's lat/lon coordinate values, used
    to key cached region masks to the exact grid they were built for. Two
    calls with the same lat/lon values get the same signature; a
    different resolution or domain gets a different one.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(lat, dtype="float64").tobytes())
    h.update(np.ascontiguousarray(lon, dtype="float64").tobytes())
    return h.hexdigest()[:16]


def get_or_build_region_mask(
    lat,
    lon,
    shapefile,
    id_col,
    cache_dir,
    name_col=None,
    simplify_tolerance=None,
    n_jobs=1,
    force_rebuild=False,
):
    """
    Load a cached region mask for this exact combination of grid,
    shapefile, and build parameters if one already exists in cache_dir;
    otherwise build it with build_region_mask and cache the result.

    The cache key covers everything that affects the resulting mask
    (grid coordinates, shapefile, id_col, simplify_tolerance) - changing
    any of these produces a different cache entry rather than reusing a
    stale one built under different settings.

    Input
    -------
    lat, lon: array-like
        The target grid's coordinates - same as build_region_mask.
    shapefile: str
        Path to the region shapefile - same as build_region_mask.
    id_col, name_col, simplify_tolerance, n_jobs:
        Passed through to build_region_mask on a cache miss, and (except
        name_col and n_jobs, which don't affect the resulting values)
        used as part of the cache key.
    cache_dir: str
        Directory where cached mask NetCDFs are stored/looked up. Should
        be shared/scratch storage sized for potentially large files at
        fine grid resolutions, not version control.
    force_rebuild: bool
        If True, rebuild and overwrite the cache even if a matching entry
        already exists.

    Output
    -------
    region_mask: xarray.DataArray ; (region, lat, lon)
        Same as build_region_mask's return value, whether loaded from
        cache or freshly built.
    """
    os.makedirs(cache_dir, exist_ok=True)

    grid_sig = _grid_signature(lat, lon)
    shapefile_tag = os.path.splitext(os.path.basename(shapefile))[0]
    if simplify_tolerance is None:
        tol_tag = "auto"
    else:
        tol_tag = str(simplify_tolerance).replace(".", "p")
    cache_file = os.path.join(
        cache_dir, f"region_mask_{shapefile_tag}_{id_col}_{tol_tag}_{grid_sig}.nc"
    )

    if os.path.exists(cache_file) and not force_rebuild:
        return load_region_mask(cache_file)

    region_mask = build_region_mask(
        lat, lon, shapefile, id_col,
        name_col=name_col, simplify_tolerance=simplify_tolerance, n_jobs=n_jobs,
    )
    region_mask.to_dataset(name="mask").to_netcdf(cache_file)
    return region_mask


def cell_area_km2(lat, lon):
    """
    Physical area of each grid cell in km^2, for a regular lat-lon grid.

    Distinct from build_lat_weights: cosine-latitude alone gives the
    correct relative weighting between cells at different latitudes, but
    is missing the constant scale factor (Earth radius squared times the
    cell's angular width in both directions) that converts it into an
    actual area. That constant cancels out when weights are normalised
    (as in aggregate_region's "mean"), so it never mattered there, but it
    does not cancel for an un-normalised total ("sum"), which is why
    aggregate_region uses this function specifically for "sum" rather
    than the passed-in lat_weights.

    Input
    -------
    lat: array-like
        1D array of latitude values, regular spacing.
    lon: array-like
        1D array of longitude values, regular spacing.

    Output
    -------
    area: xarray.DataArray ; (lat, lon)
        Cell area in km^2.
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    dlat = abs(lat[1] - lat[0])
    dlon = abs(lon[1] - lon[0])
    earth_radius_km = 6371.0

    area_1d = (
        earth_radius_km**2
        * np.deg2rad(dlat)
        * np.deg2rad(dlon)
        * np.cos(np.deg2rad(lat))
    ).astype("float32")
    area = np.broadcast_to(area_1d[:, None], (len(lat), len(lon)))

    return xr.DataArray(
        area,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="cell_area_km2",
    )


def build_lat_weights(lat, lon):
    """
    Generate cosine-latitude weights for a target grid - the correction
    for a regular lat-lon grid's cells covering less physical area toward
    the poles than near the equator.

    Input
    -------
    lat: array-like
        1D array of latitude values.
    lon: array-like
        1D array of longitude values (used only to broadcast to (lat, lon)).

    Output
    -------
    lat_weights: xarray.DataArray ; (lat, lon)
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    weights_1d = np.cos(np.deg2rad(lat)).astype("float32")
    weights = np.broadcast_to(weights_1d[:, None], (len(lat), len(lon)))

    return xr.DataArray(
        weights,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="lat_weights",
    )


def load_lat_weights(lat_weights_file):
    """
    Load a precomputed cosine-latitude weights file.

    Input
    -------
    lat_weights_file: str
        Path to a NetCDF file with a (lat, lon) or (latitude, longitude)
        weights variable.

    Output
    -------
    lat_weights: xarray.DataArray ; (lat, lon)
    """
    ds = xr.open_dataset(lat_weights_file)
    var_name = list(ds.data_vars)[0]
    lat_weights = ds[var_name]
    return _rename_to_lat_lon(lat_weights)


def _init_agg_worker(data, lat_weights, stat):
    """Runs once per worker process, not once per chunk, keeps the shared
    data/lat_weights/stat from being re-sent through IPC for every task."""
    global _worker_data, _worker_lat_weights, _worker_stat
    _worker_data = data
    _worker_lat_weights = lat_weights
    _worker_stat = stat


def _aggregate_chunk(region_mask_chunk):
    """Worker: same computation as the sequential path in aggregate_region,
    for whichever stat was set by _init_agg_worker, on a smaller mask
    slice."""
    data = _worker_data
    lat_weights = _worker_lat_weights
    stat = _worker_stat

    weights = None
    if stat == "mean":
        weights = region_mask_chunk
        if lat_weights is not None:
            sel_kwargs = {"lat": data.lat}
            if "lon" in lat_weights.dims:
                sel_kwargs["lon"] = data.lon
            lat_weights_aligned = lat_weights.sel(**sel_kwargs)
            weights = weights * lat_weights_aligned
    elif stat == "sum":
        area = cell_area_km2(data.lat.values, data.lon.values)
        weights = region_mask_chunk * area

    if stat == "mean":
        norm = weights.sum(dim=("lat", "lon"))
        return (data * weights).sum(dim=("lat", "lon")) / norm.where(norm != 0)
    elif stat == "sum":
        return (data * weights).sum(dim=("lat", "lon"))

    covered = region_mask_chunk > 0
    if stat == "min":
        return data.where(covered).min(dim=("lat", "lon"))
    elif stat == "max":
        return data.where(covered).max(dim=("lat", "lon"))
    elif stat == "std":
        return data.where(covered).std(dim=("lat", "lon"))


def aggregate_region(data, region_mask, lat_weights=None, stat="mean", n_jobs=1):
    """
    Reduce a gridded (time, lat, lon) indicator field to one value per
    region, using a precomputed region mask. Indicator-agnostic: works on
    whatever field is passed in. Cheap relative to build_region_mask,
    called repeatedly, once per indicator file; region_mask should
    already be built (see build_region_mask / get_or_build_region_mask).
    At native grid resolution the mask can be tens of GB though, so use
    n_jobs rather than assuming this step is instant.

    Input
    -------
    data: xarray.DataArray or xarray.Dataset ; (time, lat, lon)
        Already-computed indicator field.
    region_mask: xarray.DataArray ; (region, lat, lon)
        Fractional coverage mask from load_region_mask/build_region_mask.
        Also accepts a binary/hard-assignment mask - this function only
        multiplies and sums, so either representation works, though
        results will differ at coastlines/small regions depending on
        which was used to build region_mask.
    lat_weights: xarray.DataArray or None ; (lat, lon)
        Cosine-latitude weights from build_lat_weights/load_lat_weights.
        Only meaningful if data is on a regular lat-lon grid; omit for
        equal-area grids to avoid double-counting area weighting. Only
        affects stat="mean" - stat="sum" always computes real physical
        cell area itself (see cell_area_km2), since a true total needs
        actual area, not a relative weighting, and this parameter is
        ignored in that case.
    stat: str
        One of "mean", "sum", "min", "max", "std". Default "mean".
    n_jobs: int
        Worker processes for parallel per-region-chunk computation.
        Default 1 (sequential). At native grid resolution the mask itself
        can be tens of GB, so this is memory-bandwidth bound rather than
        compute bound; splitting the region dimension across cores gives
        a close-to-linear speedup, the same principle as
        build_region_mask's tiling, applied to a differently-shaped
        computation.

    Output
    -------
    out: xarray.DataArray or xarray.Dataset ; (time, region)
        Aggregated field, one value per region per timestep.

    Notes
    -------
    "mean" is a weighted average (fractional coverage x cosine-latitude,
    normalised to sum to 1 per region). "sum" is fractional coverage
    weighted by true physical cell area (see cell_area_km2), not the
    passed-in lat_weights, a cell 30% inside a region contributes 30%
    of its area to the total, and the result is in the field's original
    units times km^2, an actual physical total. "min"/"max"/"std" mask
    to cells with any coverage and reduce unweighted.

    Spatial aggregation reduces (lat, lon) but does not touch the time
    dimension, the output is still a full time series, one value per
    region per original timestep. If a temporal reduction (e.g. an annual
    mean) is also wanted, apply it as an explicit, separate step.
    """
    if get_type(data) == "Dataset":
        return xr.Dataset(
            {
                var: aggregate_region(data[var], region_mask, lat_weights, stat, n_jobs)
                for var in data.data_vars
            }
        )

    assert stat in SUPPORTED_STATS, f'"{stat}" is not a supported stat. Choose from {SUPPORTED_STATS}.'

    # Mask and data can come from different pipelines and differ by
    # floating-point noise. Round to a precision far below any grid spacing.
    data = data.assign_coords(lat=data.lat.round(6), lon=data.lon.round(6))
    region_mask = region_mask.assign_coords(
        lat=region_mask.lat.round(6), lon=region_mask.lon.round(6)
    )
    if lat_weights is not None:
        lw_coords = {"lat": lat_weights.lat.round(6)}
        if "lon" in lat_weights.dims:
            lw_coords["lon"] = lat_weights.lon.round(6)
        lat_weights = lat_weights.assign_coords(**lw_coords)
    data, region_mask = xr.align(data, region_mask, join="inner")
    # Real indicator files are commonly float64 while the mask is float32;
    # multiplying the two silently upcasts every intermediate array to
    # float64, doubling memory traffic for no precision benefit at these
    # magnitudes. Cast to match the mask's dtype before any arithmetic.
    if data.dtype != region_mask.dtype:
        data = data.astype(region_mask.dtype)

    if n_jobs == 1:
        weights = None
        if stat == "mean":
            weights = region_mask
            if lat_weights is not None:
                # lat_weights may be 1D (lat only, e.g. CDS's LATW-coeff file)
                # or 2D (lat, lon, e.g. PECD's/our own build_lat_weights) -
                # only select on dims it actually has; xarray broadcasts a 1D
                # (lat,) array against the (region, lat, lon) weights fine.
                sel_kwargs = {"lat": data.lat}
                if "lon" in lat_weights.dims:
                    sel_kwargs["lon"] = data.lon
                lat_weights_aligned = lat_weights.sel(**sel_kwargs)
                weights = weights * lat_weights_aligned
        elif stat == "sum":
            # Real physical area, not the passed-in lat_weights - see
            # cell_area_km2's docstring for why sum needs this and mean
            # does not.
            area = cell_area_km2(data.lat.values, data.lon.values)
            weights = region_mask * area

        if stat == "mean":
            norm = weights.sum(dim=("lat", "lon"))
            out = (data * weights).sum(dim=("lat", "lon")) / norm.where(norm != 0)
        elif stat == "sum":
            out = (data * weights).sum(dim=("lat", "lon"))
        else:
            covered = region_mask > 0
            if stat == "min":
                out = data.where(covered).min(dim=("lat", "lon"))
            elif stat == "max":
                out = data.where(covered).max(dim=("lat", "lon"))
            elif stat == "std":
                out = data.where(covered).std(dim=("lat", "lon"))
    else:
        import concurrent.futures

        n_regions = region_mask.sizes["region"]
        chunk_idx = np.array_split(np.arange(n_regions), min(n_jobs, n_regions))
        chunks = [region_mask.isel(region=idx) for idx in chunk_idx]

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_jobs, initializer=_init_agg_worker, initargs=(data, lat_weights, stat),
        ) as executor:
            results = list(executor.map(_aggregate_chunk, chunks))

        out = xr.concat(results, dim="region")

    out.name = data.name
    out.attrs = data.attrs

    return out

