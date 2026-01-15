"""
This sctipt creates the metadata in the structure that is requested in the DataLake
(see https://destine-data-lake-docs.data.destination-earth.eu/en/latest/dedl-discovery-and-data-access/User-Generated-Data/Promote-user-data-to-become-DestinE-data/Promote-user-data-to-become-DestinE-data.html)
"""

import os
from datetime import datetime
from shapely.geometry import box
import pystac
import xarray as xr

def extract_datetime_from_filename(filename):
    """Extract datetime from a NetCDF filename."""
    try:
        # Split the filename by underscores and extract the last part before '.nc'
        date_str = filename.split("_")[-1].replace(".nc", "")

        # Trim any nanosecond precision beyond the microsecond precision (6 digits)
        if len(date_str.split(".")[-1]) > 6:
            date_str = date_str.rsplit(".", 1)[0]  # Keep only microseconds precision

        # Parse the datetime string (now with microsecond precision)
        extracted_datetime = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
        return extracted_datetime
    except ValueError as e:
        print(f"Failed to parse datetime from filename: {filename} with error: {e}")
        return None


def get_spatial_extent(dataset):
    """Extract spatial extent (bounding box) from a NetCDF dataset."""
    lat_min, lat_max = dataset.lat.min().item(), dataset.lat.max().item()
    lon_min, lon_max = dataset.lon.min().item(), dataset.lon.max().item()
    return [lon_min, lat_min, lon_max, lat_max]


def create_stac_collection(output_path, spatial_extent, temporal_extent):
    """Create a STAC Collection and save it."""
    collection = pystac.Collection(
        id="netcdf-data-collection",
        description="A collection of NetCDF-based STAC Items for the energy onshore use case of the Climate Adaptation Digital Twin from DestinE.",
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent([spatial_extent]),
            temporal=pystac.TemporalExtent([temporal_extent]),
        ),
        license="CC-BY-4.0",
        title="NetCDF Data Collection",
        keywords=["NetCDF", "Climate", "Energy", "Geospatial"],
        providers=[
            pystac.Provider(
                name="Barcelona Supercomputing Center (BSC)",
                roles=["producer", "licensor"],
                url="https://www.bsc.es/",
            )
        ],
    )
    collection.save_object(dest_href=output_path)
    print(f"Collection saved to {output_path}")


def create_stac_item(netcdf_path, output_path):
    """Create a STAC Item for a given NetCDF file and save it."""
    dataset = xr.open_dataset(netcdf_path)

    # Extract metadata
    filename = os.path.basename(netcdf_path)
    datetime_obj = extract_datetime_from_filename(filename)
    spatial_extent = get_spatial_extent(dataset)
    geometry = box(*spatial_extent).__geo_interface__

    # Create STAC Item
    item = pystac.Item(
        id=filename.replace(".nc", ""),
        geometry=geometry,
        bbox=spatial_extent,
        datetime=datetime_obj,
        properties={},
    )

    # Add asset
    item.add_asset(
        key="data",
        asset=pystac.Asset(
            href=netcdf_path,
            media_type="application/x-netcdf",
            title="NetCDF File",
            roles=["data"],
        ),
    )

    # Save STAC Item
    item.save_object(dest_href=output_path)
    print(f"STAC Item saved to {output_path}")


if __name__ == "__main__":
    # Directory structure
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data")
    metadata_dir = os.path.join(base_dir, "metadata")
    items_dir = os.path.join(metadata_dir, "items")

    # Ensure directories exist
    os.makedirs(items_dir, exist_ok=True)

    # Initialize spatial and temporal extents
    all_bboxes = []
    all_datetimes = []

    # Process each NetCDF file
    for file in os.listdir(data_dir):
        if file.endswith(".nc"):
            netcdf_path = os.path.join(data_dir, file)
            item_path = os.path.join(items_dir, f"{file.replace('.nc', '.json')}")

            # Create STAC Item
            dataset = xr.open_dataset(netcdf_path)
            bbox = get_spatial_extent(dataset)
            all_bboxes.append(bbox)

            datetime_obj = extract_datetime_from_filename(file)
            if datetime_obj:
                all_datetimes.append(datetime_obj)

            create_stac_item(netcdf_path, item_path)

    # Determine collection extents
    spatial_extent = [
        min([b[0] for b in all_bboxes]),
        min([b[1] for b in all_bboxes]),
        max([b[2] for b in all_bboxes]),
        max([b[3] for b in all_bboxes]),
    ]
    temporal_extent = [
        min(all_datetimes),
        max(all_datetimes),
    ]

    print(temporal_extent)
    # Create STAC Collection
    collection_path = os.path.join(metadata_dir, "collection.json")
    create_stac_collection(collection_path, spatial_extent, temporal_extent)

