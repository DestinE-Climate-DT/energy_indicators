#!/usr/bin/env python3
"""
Validate our own fractional coverage country mask against the CDS global
reference mask (sis-energy-global-reanalysis), by running the same real
indicator field through both pipelines and comparing per country results.

Both sides use the same underlying field, so any disagreement reflects
mask precision, not a difference in the climate data itself.

Steps:
  1. Load a real, native resolution indicator field.
  2. Aggregate it with our mask, at native resolution, giving "ours".
  3. Regrid (interpolate) the same field onto the CDS mask's coarser grid.
  4. Aggregate the regridded field with the CDS mask, giving "reference".
  5. Match the two by country (ISO3 vs ISO2, via a crosswalk built from
     Natural Earth's own columns) and compare.
  6. Also plot the raw indicator field, with country borders, as a
     sanity check.
"""

import time as _time
import argparse
import os
import sys
import types
import importlib.util

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd

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

INDICATOR_VAR = "ws"

# CDS global reference mask and latitude weights, from the
# sis-energy-global-reanalysis dataset, "Weights and masks" section:
# https://cds.climate.copernicus.eu/datasets/sis-energy-global-reanalysis
# Download "Country aggregation mask" and "Latitude weighting coefficients".
CDS_MASK_FILE = os.path.expanduser("~/energy_indicators/energy_indicators/global_masks/ANCI_ADM0-mask_C3S2LOT1_025d_v1.00.nc")
CDS_LATW_FILE = os.path.expanduser("~/energy_indicators/energy_indicators/global_masks/ANCI_LATW-coeff_C3S2LOT1_025d_v1.00.nc")

# Natural Earth admin_0_countries, 1:10m, used here only for its ISO3/ISO2
# crosswalk columns:
# https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/
NATURAL_EARTH_SHAPEFILE = os.path.expanduser("~/energy_indicators/energy_indicators/global_masks/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp")

BOUNDARIES_FILE = os.path.expanduser("~/energy_indicators/energy_indicators/boundaries/global_country_boundaries.gpkg")

OUT_DIR = os.path.expanduser("~/energy_indicators_validation")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--indicator-file", required=True, help="Real indicator NetCDF to validate against.")
parser.add_argument("--our-mask-file", required=True, help="Path to our own cached region mask NetCDF.")
parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)), help="Worker processes for aggregate_region. Defaults to $SLURM_CPUS_PER_TASK if set, else 1.")
args = parser.parse_args()

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading indicator field...", flush=True)
ds = xr.open_dataset(args.indicator_file)
data = ds[INDICATOR_VAR].isel(time=0) if "time" in ds[INDICATOR_VAR].dims else ds[INDICATOR_VAR]

print("Sanity plot of the raw field...", flush=True)
data_plot = data.assign_coords(lon=(((data.lon + 180) % 360) - 180)).sortby("lon")
boundaries = gpd.read_file(BOUNDARIES_FILE)

fig, ax = plt.subplots(figsize=(16, 8))
data_plot.plot(ax=ax, cmap="YlOrRd", add_colorbar=True)
boundaries.boundary.plot(ax=ax, color="black", linewidth=0.3)
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_title(f"{INDICATOR_VAR}, raw field, native resolution")
fig.savefig(os.path.join(OUT_DIR, f"{INDICATOR_VAR}_raw_field.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print("Loading our mask and building lat weights at native resolution...", flush=True)
our_mask = agg.load_region_mask(args.our_mask_file)
our_lat_weights = agg.build_lat_weights(our_mask.lat.values, our_mask.lon.values)

print("Aggregating with our mask (native resolution)...", flush=True)
_t0 = _time.time()
ours = agg.aggregate_region(data, our_mask, our_lat_weights, stat="mean", n_jobs=args.n_jobs)
print(f"  took {_time.time() - _t0:.1f}s", flush=True)
ours_df = ours.to_dataframe(name="value").reset_index()[["region", "value"]]
ours_df = ours_df.rename(columns={"value": "ours"})

print("Plotting aggregated per country result...", flush=True)
choropleth = boundaries.merge(ours_df, on="region", how="left")

fig, ax = plt.subplots(figsize=(16, 8))
choropleth.plot(
    column="ours", ax=ax, cmap="YlOrRd", legend=True,
    edgecolor="black", linewidth=0.2,
    missing_kwds={"color": "lightgrey", "label": "no data"},
)
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_title(f"{INDICATOR_VAR}, aggregated (mean) per country (our native resolution mask)")
fig.savefig(os.path.join(OUT_DIR, f"{INDICATOR_VAR}_aggregated_per_country.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print("Loading CDS reference mask and lat weights...", flush=True)
cds_mask = agg.load_region_mask(CDS_MASK_FILE)
cds_lat_weights = agg.load_lat_weights(CDS_LATW_FILE)

print("Regridding indicator field onto the CDS grid (linear interpolation)...", flush=True)
data_on_cds_grid = data.interp(lat=cds_mask.lat.values, lon=cds_mask.lon.values, method="linear")

print("Aggregating with CDS reference mask...", flush=True)
_t0 = _time.time()
reference = agg.aggregate_region(data_on_cds_grid, cds_mask, cds_lat_weights, stat="mean")
print(f"  took {_time.time() - _t0:.1f}s", flush=True)
reference_df = reference.to_dataframe(name="value").reset_index()[["region", "value"]]
reference_df = reference_df.rename(columns={"value": "reference"})

print("Building ISO3 to ISO2 crosswalk from Natural Earth...", flush=True)
ne = gpd.read_file(NATURAL_EARTH_SHAPEFILE)
ne["iso_a2_resolved"] = ne["ISO_A2"].where(ne["ISO_A2"] != "-99", ne["ISO_A2_EH"])
# Dependencies and similar non sovereign entities share their parent's
# ISO2 code, which would otherwise wrongly match them against the
# parent's reference value. Keep only genuinely sovereign countries.
ne_sovereign = ne[ne["TYPE"].isin(["Sovereign country", "Country"])]
crosswalk = ne_sovereign[["ADM0_A3", "iso_a2_resolved"]].rename(
    columns={"ADM0_A3": "region", "iso_a2_resolved": "region_iso2"}
)
crosswalk = crosswalk[crosswalk["region_iso2"] != "-99"]

print("Merging and comparing...", flush=True)
n_ours_before_crosswalk = len(ours_df)
ours_df = ours_df.merge(crosswalk, on="region", how="inner")
n_dropped_non_sovereign = n_ours_before_crosswalk - len(ours_df)
n_ours_after_crosswalk = len(ours_df)
comparison = ours_df.merge(
    reference_df.rename(columns={"region": "region_iso2"}), on="region_iso2", how="inner"
)
n_dropped_no_cds_match = n_ours_after_crosswalk - len(comparison)
n_dupes = comparison["region_iso2"].duplicated(keep=False).sum()
if n_dupes:
    print(f"WARNING: {n_dupes} still ambiguous ISO2 matches after sovereignty filtering, check manually.")

comparison["abs_diff"] = comparison["ours"] - comparison["reference"]
# Percent difference is only meaningful where the reference value is
# itself meaningfully non zero. Report it only over rows above a floor,
# and use median/max rather than mean even then.
REFERENCE_FLOOR = comparison["reference"].abs().quantile(0.5) * 0.1
meaningful = comparison[comparison["reference"].abs() > REFERENCE_FLOOR].copy()
meaningful["pct_diff"] = 100 * meaningful["abs_diff"] / meaningful["reference"]

comparison.to_csv(os.path.join(OUT_DIR, f"{INDICATOR_VAR}_validation_comparison.csv"), index=False)

print(f"\nMatched {len(comparison)} countries ({len(meaningful)} with a meaningful, non near zero reference value).")
print(f"Dropped {n_dropped_non_sovereign} non sovereign territories (dependencies etc, ambiguous ISO2 mapping).")
print(f"Dropped {n_dropped_no_cds_match} further countries with no matching entry in the CDS reference set.")
print(f"\nAbsolute difference (all {len(comparison)} matched countries):")
print("  median |abs diff|:", comparison["abs_diff"].abs().median())
print("  max |abs diff|:", comparison["abs_diff"].abs().max())
print(f"\nPercent difference (only the {len(meaningful)} countries with reference > {REFERENCE_FLOOR:.4g}):")
print("  median |% diff|:", meaningful["pct_diff"].abs().median())
print("  max |% diff|:", meaningful["pct_diff"].abs().max())
print("\nTop 15 largest absolute discrepancies (all matched countries):")
print(comparison.sort_values("abs_diff", key=abs, ascending=False)
      [["region", "region_iso2", "ours", "reference", "abs_diff"]].head(15).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(comparison["reference"], comparison["ours"], alpha=0.6)
lims = [comparison[["ours", "reference"]].min().min(), comparison[["ours", "reference"]].max().max()]
ax.plot(lims, lims, "k--", alpha=0.5, label="1:1")
ax.set_xlabel(f"CDS reference mask, {INDICATOR_VAR}")
ax.set_ylabel(f"Our native resolution mask, {INDICATOR_VAR}")
ax.set_title(f"{INDICATOR_VAR}: per country comparison, {len(comparison)} countries")
ax.legend()
fig.savefig(os.path.join(OUT_DIR, f"{INDICATOR_VAR}_validation_scatter.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nOutputs written to {OUT_DIR}")
