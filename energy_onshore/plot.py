"""
Script to define the functions to plot the results.
"""
# Plotting libraries


# Matplotlib
def load_plot_libs():
    """
    Load essential libraries for plotting and geospatial visualizations.

    This function imports several key libraries required for creating and customizing
    advanced visualizations, including general plotting tools, geospatial visualization
    utilities, and oceanographic colormaps. These libraries facilitate the creation of
    maps, handling of colormaps, and manipulation of geospatial data.

    Libraries Imported
    -------------------
    - `os`: Provides operating system interface for file and directory manipulation.
    - `xarray`: For handling and analyzing multi-dimensional data arrays, often used with NetCDF files.
    - `numpy`: A fundamental package for numerical computation.
    - `pandas`: A library for data manipulation and analysis, particularly for structured data.
    - `matplotlib`:
        - `pyplot`: A state-based interface to matplotlib for creating plots and figures.
        - `gridspec`: A module for managing complex subplot layouts.
        - `ScalarFormatter`: Formatter for axis ticks using scalar values.
        - `patches`: For adding shapes (e.g., circles, rectangles) to plots.
        - `colors`: For manipulating colormap properties.
        - `ticker`: For customizing tick marks and labels on plots.
    - `cartopy`: For geospatial data visualization.
        - `crs`: Handles cartographic projections and coordinate reference systems.
        - `feature`: Provides access to map features like coastlines, borders, rivers, etc.
        - `add_cyclic_point`: A utility to handle cyclic data (e.g., longitude wrapping in global datasets).
    - `cmocean`: A library of perceptually uniform colormaps, ideal for oceanographic and climate data.

    Notes
    -----
    This function does not return any values but ensures all necessary libraries
    are imported and ready for use in subsequent visualizations. It is recommended
    to call this function at the beginning of scripts that involve geospatial
    or scientific data visualization.

    Returns
    -------
    None
        The function performs library imports but does not return any value.

    Examples
    --------
    Example usage in a script:
    >>> load_plot_libs()
    >>> # Proceed with plotting code, e.g., matplotlib or cartopy visualizations
    """
    import os  # Ensures file/directory operations are supported
    import xarray as xr
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.util import add_cyclic_point
    import cmocean as cmo


# Define colormaps
def define_colormaps():
    """
    Define colormaps for the other ploting functions.
    """
    cmap_cmocean = cmo.cm.tempo
    cmap_lwe = cmo.cm.tempo
    cmap_hwe = cmo.cm.amp


# Updated plot_map function
def plot_map(
    data,
    xmin,
    xmax,
    colormap,
    label,
    title,
    delta=12,
    projection="robinson",
    type="pcolormesh",
    extend="neither",
    ax=None,
    output_file=None,
):
    """
    Plots a geographical map of a variable using a specified projection, colormap, and plotting style.

    Parameters
    ----------
    data : xarray.DataArray
        The data to be plotted. Must contain latitude (`lat`) and longitude (`lon`) dimensions.
    xmin : float
        Minimum value for the color scale.
    xmax : float
        Maximum value for the color scale.
    colormap : str or matplotlib.colors.Colormap
        Colormap to use for the plot.
    label : str
        Label for the colorbar.
    title : str
        Title for the plot.
    delta : int, optional, default=12
        The increment for contour levels when `type='contourf'`. Not used for `pcolormesh`.
    projection : str, optional, default='robinson'
        The map projection. Currently supports 'robinson' by default.
    type : str, optional, default='pcolormesh'
        Type of plot to generate. Options are:
        - `'pcolormesh'`: Creates a pseudocolor plot.
        - `'contourf'`: Creates filled contour plots with optional contour lines.
    extend : str, optional, default='neither'
        Determines how the colorbar handles out-of-range values. Options:
        - `'neither'`: No extensions.
        - `'both'`: Extend at both ends.
        - `'min'`: Extend at the lower end.
        - `'max'`: Extend at the upper end.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Existing matplotlib axis object. If `None`, a new figure and axis will be created.
    output_file : str, optional
        File path to save the plot. If `None`, the plot will not be saved.

    Notes
    -----
    - The map uses a Robinson projection by default, with coastlines and land features added.
    - If the `data` spans the 0-360 longitude range, the function automatically handles longitude wrapping using `add_cyclic_point`.
    - Gridlines and labels are styled for readability.

    Returns
    -------
    None
        The function does not return any values. It displays or saves the plot, depending on the parameters.

    Examples
    --------
    Example usage with xarray DataArray:
    >>> plot_map(data=my_data, xmin=0, xmax=100, colormap='viridis', label='Temperature (°C)',
                 title='Global Temperature Map', type='pcolormesh', output_file='map.png')

    """
    proj = ccrs.Robinson()

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 10), subplot_kw={"projection": proj})

    lats = data.lat.values
    lons = data.lon.values
    var = data.values

    var, lons = add_cyclic_point(var, coord=lons)

    lon2d, lat2d = np.meshgrid(lons, lats)

    if type == "pcolormesh":
        cs = ax.pcolormesh(
            lon2d,
            lat2d,
            var,
            transform=ccrs.PlateCarree(),
            cmap=colormap,
            vmin=xmin,
            vmax=xmax,
        )
    elif type == "contourf":
        clevs = np.linspace(xmin, xmax, 256)
        cs = ax.contourf(
            lon2d,
            lat2d,
            var,
            levels=clevs,
            cmap=colormap,
            extend=extend,
            transform=ccrs.PlateCarree(),
        )
        ax.contour(
            lon2d,
            lat2d,
            var,
            clevs,
            colors="k",
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
        )

    ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="whitesmoke")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.75, edgecolor="black")

    gl = ax.gridlines(
        draw_labels=True, linewidth=1, color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 12}
    gl.ylabel_style = {"size": 12}

    cbar = plt.colorbar(
        cs, ax=ax, orientation="horizontal", pad=0.05, extend=extend, shrink=0.7
    )
    cbar.set_label(label)
    cbar.ax.tick_params(labelsize=12)

    plt.title(title)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_file}")

    plt.close()


# Define the plot_cf function
def plot_cf(file_paths, output_directory, iec_class='i'):
    """
    Plot capacity factor indicators.

    Parameters
    ----------
    input_files : dict
        Dictionary containing input file paths for 'cfX'.
        Example: {'cdd_acc': 'path_to_cdd_file.nc', 'hdd_acc': 'path_to_hdd_file.nc'}
    output_directory : str
        Path to save the output plots.
    """

    os.makedirs(output_directory, exist_ok=True)
    for category, file_path in file_paths.items():
        try:
            ds = xr.open_dataset(file_path)
            cf = ds[f"cf_{iec_class}.lower()"]
            daily_mean = cf.resample(time="1D").mean()
            first_day_date = pd.to_datetime(daily_mean["time"].values[0]).strftime(
                "%Y-%m-%d"
            )
            output_file = os.path.join(
                output_directory,
                f"{category.upper().replace('_', '')}_daily_mean_{first_day_date}.png",
            )
            plot_map(
                data=daily_mean.isel(time=0),
                xmin=0,
                xmax=1,
                colormap=cmap_cmocean,
                label="Capacity Factor (%)",
                title=f'{category.upper().replace("_", "")} - Daily Mean Capacity Factor ({first_day_date})',
                output_file=output_file,
            )
        except Exception as e:
            print(f"Error processing {category}: {e}")


# Define the plot_lwe_hwe function
def plot_lwe_hwe(input_files, output_directory):
    """
    Plot accumulated cooling degree days (CDD) and heating degree days (HDD).

    Parameters
    ----------
    input_files : dict
        Dictionary containing input file paths for 'cdd_acc' and 'hdd_acc'.
        Example: {'cdd_acc': 'path_to_cdd_file.nc', 'hdd_acc': 'path_to_hdd_file.nc'}
    output_directory : str
        Path to save the output plots.
    """
    os.makedirs(output_directory, exist_ok=True)
    for event_type, file_path in input_files.items():
        try:
            ds = xr.open_dataset(file_path)
            event_data = ds[event_type]
            perc_occur = (event_data / 24) * 100
            first_day_date = file_path.split("/")[-1].split("_")[1].split("T")[0]
            colormap = cmap_lwe if event_type == "lwe" else cmap_hwe
            output_file = os.path.join(
                output_directory,
                f"{event_type.upper()}_perc_occur_{first_day_date}.png",
            )
            plot_map(
                data=perc_occur,
                xmin=0,
                xmax=100,
                colormap=colormap,
                label=f"{event_type.upper()} Percentage Occurrence (%)",
                title=f"{event_type.upper()} - Daily Accumulated Percentage Occurrence ({first_day_date})",
                output_file=output_file,
            )
        except Exception as e:
            print(f"Error processing {event_type}: {e}")


# define plotting for cdd and hdd days
def plot_degree_days(input_files, output_directory):
    """
    Plot accumulated cooling degree days (CDD) and heating degree days (HDD).

    Parameters
    ----------
    input_files : dict
        Dictionary containing input file paths for 'cdd_acc' and 'hdd_acc'.
        Example: {'cdd_acc': 'path_to_cdd_file.nc', 'hdd_acc': 'path_to_hdd_file.nc'}
    output_directory : str
        Path to save the output plots.
    """
    os.makedirs(output_directory, exist_ok=True)

    for variable, file_path in input_files.items():
        print(f"Processing file: {file_path} for {variable.upper()}")

        # Load the dataset
        try:
            ds = xr.open_dataset(file_path)
            print(f"{variable.upper()} NetCDF file loaded successfully.")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue

        # Extract data and variable
        data = ds[variable]
        data.attrs["units"] = "°C"  # Ensure the units are consistent for the colorbar

        # Get data min and adjust data max to the nearest multiple of 5
        data_min = data.min().values
        data_max = data.max().values
        data_max_adjusted = (
            int(data_max / 5) + 1
        ) * 5  # Round up to the nearest multiple of 5

        # Extract the date from the file name
        try:
            date_str = file_path.split("_")[2].split("T")[
                0
            ]  # Extract date from file name
            first_day_date = pd.Timestamp(date_str)
        except Exception as e:
            print(f"Error extracting date from file name for {variable}: {e}")
            first_day_date = None

        # Format variable name for title and file name
        formatted_variable = variable.replace("_acc", "").upper()
        if first_day_date is not None:
            title_text = f"{formatted_variable} - Daily Accumulated {first_day_date.strftime('%Y-%m-%d')}"
            output_file_name = f"{formatted_variable}_Daily_Accumulated_{first_day_date.strftime('%Y-%m-%d')}.png"
        else:
            title_text = f"{formatted_variable} - Daily Accumulated (Unknown Date)"
            output_file_name = (
                f"{formatted_variable}_Daily_Accumulated_Unknown_Date.png"
            )

        # Define the colormap
        colormap = "rainbow"

        # Define the output file path
        output_file = os.path.join(output_directory, output_file_name)

        # Plot the data
        plot_map(
            data=data,
            xmin=data_min,
            xmax=data_max_adjusted,  # Adjusted max value
            colormap=colormap,
            label=f"{formatted_variable} Accumulated (°C)",
            title=title_text,  # Updated title with variable and date
            delta=10,  # Adjust the number of ticks dynamically via delta
            projection="robinson",
            type="pcolormesh",
            extend="max",  # Extend only for max
            output_file=output_file,
        )
