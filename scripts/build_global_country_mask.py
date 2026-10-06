#!/usr/bin/env python3
"""
Build the global (NUTS plus Natural Earth ADMIN0) country level fractional
coverage region mask, at the real Climate DT output grid.

get_or_build_region_mask caches the result, so re running this against a
grid already built just loads the cached file instead of rebuilding.

Loads aggregation.py directly rather than via a normal package import, to
avoid pulling in energy_indicators/__init__.py's other dependencies
(wind.py/solar.py, which need one_pass).
"""

import argparse
import os
import sys
import time
import types
import importlib.util

import xarray as xr

REPO_ROOT = os.path.expanduser("~/energy_indicators")
PACKAGE_DIR = os.path.join(REPO_ROOT, "energy_indicators")

pkg = types.ModuleType("energy_indicators")
pkg.__path__ = [PACKAGE_DIR]
sys.modules["energy_indicators"] = pkg

core_spec = importlib.util.spec_from_file_location(
    "energy_indicators.core", os.path.join(PACKAGE_DIR, "core.py")
)
core = importlib.util.module_from_spec(core_spec)
sys.modules["energy_indicators.core"] = core
core_spec.loader.exec_module(core)

agg_spec = importlib.util.spec_from_file_location(
    "energy_indicators.aggregation", os.path.join(PACKAGE_DIR, "aggregation.py")
)
agg = importlib.util.module_from_spec(agg_spec)
agg.__package__ = "energy_indicators"
sys.modules["energy_indicators.aggregation"] = agg
agg_spec.loader.exec_module(agg)

get_or_build_region_mask = agg.get_or_build_region_mask

N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

MERGED_SHAPEFILE = os.path.join(PACKAGE_DIR, "boundaries", "global_country_boundaries.gpkg")
CACHE_DIR = "/gpfs/projects/ehpc01/applications/energy_indicators_masks"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--reference-grid-file",
    required=True,
    help="Real indicator or raw data NetCDF, used to read the lat/lon grid.",
)
args = parser.parse_args()

print(f"Using {N_JOBS} worker processes")

ds = xr.open_dataset(args.reference_grid_file)
lat = ds["lat"].values
lon = ds["lon"].values
print(f"Reference grid: {len(lat)} x {len(lon)} = {len(lat) * len(lon):,} cells")

t0 = time.time()
region_mask = get_or_build_region_mask(
    lat,
    lon,
    shapefile=MERGED_SHAPEFILE,
    id_col="region",
    name_col="region_name",
    cache_dir=CACHE_DIR,
    n_jobs=N_JOBS,
)
t1 = time.time()

print(f"Done in {t1 - t0:.1f}s ({(t1 - t0) / 60:.1f} min)")
print(f"region_mask shape: {region_mask.shape}")
print(f"Cached under: {CACHE_DIR}")
