import numpy as np
import matplotlib.pyplot as plt
from helper import *
import os
import pandas as pd
 

def visualize_channel_list(channel_realization_list: list):
    """
    Visualize channel characteristics over time and create a GIF.

    Parameters:
    - channel_realization_list (list): List containing channel characteristics data as pandas Series at different time steps.
    """

    images = []
    save_path = os.getcwd()

    for index, time_series in enumerate(channel_realization_list):
        current_time = time_series.name  # Extract the current time from Series name
        a = time_series.values
        tau = time_series.index.values

        t = tau / 1e-9  # Scale to ns
        a_abs = np.abs(a)
        a_max = np.max(a_abs)
        t = np.reshape(t, [-1])
        a_abs = np.reshape(a_abs, [-1])

        fig, axes = plt.subplots(1, 1, figsize=(12, 4))
        # linear scale in time
        axes.set_title(f"Channel impulse response realization (Time: {current_time} s)")
        axes.stem(t, a_abs)
        #axes.set_xlim([-0.1, np.max(t) * 1.1])
        axes.set_xlim([-0.1, 42])
        axes.set_xlim([5.5, 10])
        #axes.set_ylim([-2e-12, a_max * 1.1])
        axes.set_ylim([-2e-12, 12e-6])
        axes.set_xlabel(r"$\tau$ [ns]")
        axes.set_ylabel(r"$|a|$")
        #axes.set_xticks(np.arange(0, np.max(t) * 1.1, 2))
        axes.set_xticks(np.arange(0, 40, 1))
        

        # subplots do not overlap
        plt.tight_layout()

        # Save the figure as an image
        image_filename = os.path.join(save_path, f"figure_{index}.png")
        plt.savefig(image_filename)
        plt.close()

        images.append(image_filename)

    gif_int = create_gif_from_channel(images)

    return gif_int


def visualize_channel_df(channel_magnitude_df: pd.DataFrame) -> pd.DataFrame:

    images = []
    save_path = os.getcwd()

    for column in channel_magnitude_df.columns:

        fig, axes = plt.subplots(1, 1, figsize=(12, 4))
        axes.set_title(f"Noisy quantized Channel impulse response realization (Time: {column} s)")
        axes.stem(channel_magnitude_df.index*1e9, channel_magnitude_df[column])
        axes.set_xlim([-0.1, 39])
        axes.set_ylim([-2e-12,12e-5])
        axes.set_xlabel(r"$\tau$ [ns]")
        axes.set_ylabel(r"$|a|$")
        axes.set_xticks(np.arange(0, 40, 1))
        # subplots do not overlap
        plt.tight_layout()
        
        # Save the figure as an image
        image_filename = os.path.join(save_path, f"figure_{column}.png")
        plt.savefig(image_filename)
        plt.close()

        images.append(image_filename)
        
    gif_int = 1
    gif_filename = f"noisy_channel_{gif_int}.gif"

    while os.path.exists(gif_filename):
        gif_int += 1
        gif_filename = f"noisy_channel_{gif_int}.gif"

    # Combine images into a GIF
    with imageio.get_writer(gif_filename, loop=800, duration=1000) as writer:
        for image in images:
            frame = imageio.imread(image)
            writer.append_data(frame)

    # Remove the temporary image files
    for image in images:
        os.remove(image)

    return None