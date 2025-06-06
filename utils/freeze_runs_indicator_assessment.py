```
Usage:  singularity exec ../../../../../../projects/ehpc01/containers/energy_onshore/energy_onshore_1.0.0.sif python3 my_python_file.py  ../../output/data/  cf_s 1990 2000    
```

import xarray as xr
import os
import glob
import pandas as pd
import argparse

def process_files(input_dir, output_dir, start_year, end_year):
    """
    Combine NetCDF files for a range of years, compute monthly means, and save the output.
    
    Parameters:
    input_dir (str): Path to the directory containing NetCDF files.
    output_dir (str): Path to the directory where the output files will be saved.
    start_year (int): The start year of the range.
    end_year (int): The end year of the range.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Loop through the range of years
    for year in range(start_year, end_year + 1):
        # Get list of NetCDF files for the specific year
        file_list = sorted(glob.glob(os.path.join(input_dir, f"cf_S_{year}-*.nc")))
        
        if not file_list:
            print(f"No files found for the year {year}.")
            continue
        
        # Open datasets and merge explicitly
        datasets = [xr.open_dataset(f, engine="netcdf4") for f in file_list]
        ds_merged = xr.concat(datasets, dim='time')
        
        # Close datasets
        for ds in datasets:
            ds.close()
        
        print(f"Files for {year} combined successfully.")
        
        # Compute monthly means
        ds_merged["time"] = pd.to_datetime(ds_merged["time"].values)
        grouped = ds_merged.groupby("time.month")
        
        for month, ds_group in grouped:
            ds_mean = ds_group.mean(dim='time', keep_attrs=True)
            output_filename = os.path.join(output_dir, f'monthly_mean_{year}_{month:02d}.nc')
            ds_mean.to_netcdf(output_filename)
            print(f"Saved: {output_filename}")
        
        print(f"Monthly mean computation completed for {year}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine NetCDF files for a range of years and compute monthly means.")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing NetCDF files.")
    parser.add_argument("output_dir", type=str, help="Path to the directory where the output files will be saved.")
    parser.add_argument("start_year", type=int, help="The start year of the range.")
    parser.add_argument("end_year", type=int, help="The end year of the range.")
    args = parser.parse_args()
    
    process_files(args.input_dir, args.output_dir, args.start_year, args.end_year)

