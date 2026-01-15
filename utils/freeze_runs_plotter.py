import xarray as xr
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from scipy.stats import linregress

def plot_timeseries(input_dir, start_year, end_year, lat, lon):
    """
    Plot time series from a specific point (lat, lon) for cf_I, cf_II, cf_III, and cf_S in the monthly mean NetCDF files.
    
    Parameters:
    input_dir (str): Path to the directory containing the monthly mean NetCDF files.
    start_year (int): The start year of the range.
    end_year (int): The end year of the range.
    lat (float): The latitude of the point.
    lon (float): The longitude of the point.
    """
    time_series_cf_I = []
    time_series_cf_II = []
    time_series_cf_III = []
    time_series_cf_S = []
    time_values = []
    
    # Loop through the range of years and process each file
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Construct the filenames for cf_I, cf_II, cf_III, and cf_S monthly mean files
            filename_cf_I = os.path.join(input_dir, f'{year}_{month:02d}_monthly_mean_cf_I.nc')
            filename_cf_II = os.path.join(input_dir, f'{year}_{month:02d}_monthly_mean_cf_II.nc')
            filename_cf_III = os.path.join(input_dir, f'{year}_{month:02d}_monthly_mean_cf_III.nc')
            filename_cf_S = os.path.join(input_dir, f'{year}_{month:02d}_monthly_mean_cf_S.nc')
            
            # Check if all files exist for cf_I, cf_II, cf_III, cf_S
            if not os.path.exists(filename_cf_I) or not os.path.exists(filename_cf_II) or not os.path.exists(filename_cf_III) or not os.path.exists(filename_cf_S):
                print(f"One or more files not found for year {year}, month {month}. Skipping...")
                continue
            
            # Open the datasets for cf_I, cf_II, cf_III, and cf_S
            ds_cf_I = xr.open_dataset(filename_cf_I)
            ds_cf_II = xr.open_dataset(filename_cf_II)
            ds_cf_III = xr.open_dataset(filename_cf_III)
            ds_cf_S = xr.open_dataset(filename_cf_S)
            
            # Get the variable (cf_I, cf_II, cf_III, cf_S) at the specific latitude and longitude
            value_cf_I = ds_cf_I['cf'].sel(lat=lat, lon=lon, method='nearest').values.item()
            value_cf_II = ds_cf_II['cf'].sel(lat=lat, lon=lon, method='nearest').values.item()
            value_cf_III = ds_cf_III['cf'].sel(lat=lat, lon=lon, method='nearest').values.item()
            value_cf_S = ds_cf_S['cf'].sel(lat=lat, lon=lon, method='nearest').values.item()
            
            # Append the values and time to the lists
            time_series_cf_I.append(value_cf_I)
            time_series_cf_II.append(value_cf_II)
            time_series_cf_III.append(value_cf_III)
            time_series_cf_S.append(value_cf_S)
            time_values.append(pd.to_datetime(f"{year}-{month:02d}-01"))
            
            # Close the datasets
            ds_cf_I.close()
            ds_cf_II.close()
            ds_cf_III.close()
            ds_cf_S.close()
    
    if not time_series_cf_I or not time_series_cf_II or not time_series_cf_III or not time_series_cf_S:
        print(f"No data found for the given coordinates: lat={lat}, lon={lon}.")
        return
    
    # Perform linear regression for each variable
    time_ordinal = [t.toordinal() for t in time_values]
    
    # Regression for cf_I
    slope_cf_I, intercept_cf_I, _, _, _ = linregress(time_ordinal, time_series_cf_I)
    regression_line_cf_I = [slope_cf_I * t + intercept_cf_I for t in time_ordinal]
    
    # Regression for cf_II
    slope_cf_II, intercept_cf_II, _, _, _ = linregress(time_ordinal, time_series_cf_II)
    regression_line_cf_II = [slope_cf_II * t + intercept_cf_II for t in time_ordinal]
    
    # Regression for cf_III
    slope_cf_III, intercept_cf_III, _, _, _ = linregress(time_ordinal, time_series_cf_III)
    regression_line_cf_III = [slope_cf_III * t + intercept_cf_III for t in time_ordinal]
    
    # Regression for cf_S
    slope_cf_S, intercept_cf_S, _, _, _ = linregress(time_ordinal, time_series_cf_S)
    regression_line_cf_S = [slope_cf_S * t + intercept_cf_S for t in time_ordinal]
    
    # Plot the time series
    plt.figure(figsize=(12, 8))
    plt.plot(time_values, time_series_cf_I, marker='.', linestyle='-',lw=0.5, color='b', label="cf_I")
    plt.plot(time_values, time_series_cf_II, marker='.', linestyle='-',lw=0.5, color='g', label="cf_II")
    plt.plot(time_values, time_series_cf_III, marker='.', linestyle='-',lw=0.5, color='m', label="cf_III")
    plt.plot(time_values, time_series_cf_S, marker='o', linestyle='-', color='c', label="cf_S")
    
    # Plot the regression lines
    plt.plot(time_values, regression_line_cf_I, linestyle='--', color='r', label="Regression Line cf_I")
    plt.plot(time_values, regression_line_cf_II, linestyle='--', color='orange', label="Regression Line cf_II")
    plt.plot(time_values, regression_line_cf_III, linestyle='--', color='purple', label="Regression Line cf_III")
    plt.plot(time_values, regression_line_cf_S, linestyle='--', color='c', label="Regression Line cf_S")
    
    # Labels and title
    plt.xlabel('Time')
    plt.ylabel('Monthly Mean')
    plt.title(f"Time Series of Monthly Means for cf_I, cf_II, cf_III, and cf_S at ({lat}, {lon}) with Regression Lines. IFS-NEMO control, 10km res")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.ylim(0.2, 1)  # Setting y-axis limits from 0.2 to 1
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot time series for a given point from the monthly mean NetCDF files for cf_I, cf_II, cf_III, and cf_S with regression lines.")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing the monthly mean NetCDF files.")
    parser.add_argument("start_year", type=int, help="The start year of the range.")
    parser.add_argument("end_year", type=int, help="The end year of the range.")
    parser.add_argument("lat", type=float, help="The latitude of the point.")
    parser.add_argument("lon", type=float, help="The longitude of the point.")
    args = parser.parse_args()
    
    plot_timeseries(args.input_dir, args.start_year, args.end_year, args.lat, args.lon)

