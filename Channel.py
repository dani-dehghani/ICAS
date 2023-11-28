import pandas as pd
import numpy as np

def channel_df(channel_realization_list: list, resolution_scale: int = 1, interpolation_method: str = 'linear') -> pd.DataFrame:
    """
    Converts channel realization list into a DataFrame.
    
    Args:
    - channel_realization_list (list): List of channel realization data (assumed to be complex numbers)
    - resolution_scale (int, optional): Scaling factor for time index, default is 1
    - interpolation_method (str, optional): Method of interpolation, default is 'linear'
    
    Returns:
    - channel_df (pd.DataFrame): Single DataFrame from the channel realization list consisting of all the realizations
    """
    series_list = []
    channel_df = pd.DataFrame()

    for idx, series in enumerate(channel_realization_list):
        series = series.groupby(series.index).sum()  # Grouping by the contributions of the floor and robot
        series.index = series.index * 1e9             # Making the index 'ns'
        series.index = series.index * resolution_scale
        series.index = pd.to_timedelta(series.index, unit='ns')
        series = series.groupby(series.index).sum()

        real_part = series.apply(lambda x: x.real)
        imaginary_part = series.apply(lambda x: x.imag)
        resampled_real_part = real_part.resample('ns').interpolate(method=interpolation_method)
        resampled_imaginary_part = imaginary_part.resample('ns').interpolate(method=interpolation_method)
        interpolated_series = resampled_real_part + 1j * resampled_imaginary_part

        interpolated_series.index = interpolated_series.index / resolution_scale
        interpolated_series.index = interpolated_series.index.total_seconds()
        interpolated_series = interpolated_series.groupby(interpolated_series.index).sum()
        interpolated_series = interpolated_series.replace('N00000000a00000000N', np.nan)
        interpolated_series = interpolated_series.dropna()
        series_list.append(interpolated_series)

    channel_df = pd.concat(series_list, axis=1)
    return channel_df

#%%

def add_noise_to_channel(channel_df: pd.DataFrame, noise_power_percent: float) -> pd.DataFrame:
    """
    Adds complex Gaussian noise to each column (realization) of the DataFrame.

    Args:
    - channel_df (pd.DataFrame): DataFrame containing complex data
    - noise_percentage (float): Percentage of the average signal power to be considered as noise power

    Returns:
    - noisy_channel_df (pd.DataFrame): DataFrame with added noise in each column
    """
    # Calculate the average power of the signal across all columns
    signal_power = np.mean(np.mean(np.abs(channel_df) ** 2))

    # Calculate the noise power as noise_power_percent of the average signal power
    noise_power = noise_power_percent * signal_power

    # Get the columns and index of the DataFrame
    columns = channel_df.columns
    
    # Generate complex Gaussian noise for each column
    noisy_columns = []
    for column in columns:
        # Generate noise for real and imaginary parts separately
        noise_real = np.random.normal(0, np.sqrt(noise_power / 2), len(channel_df))
        noise_imaginary = np.random.normal(0, np.sqrt(noise_power / 2), len(channel_df))
        noise = noise_real + 1j * noise_imaginary

        # Add noise to the column
        noisy_column = channel_df[column] + noise
        noisy_columns.append(noisy_column)

    # Create a new DataFrame with noisy columns
    noisy_channel_df = pd.DataFrame(noisy_columns).T
    noisy_channel_df.columns = columns

    return noisy_channel_df

#%%