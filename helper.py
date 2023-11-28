#%%
import os
import imageio
import re
from IPython.display import display, Image
import numpy as np
import mitsuba as mi
import drjit as dr
from typing import List, Tuple, Optional
from scipy.interpolate import CubicSpline
import pandas as pd
import trimesh

def create_gif_from_movement() -> Optional[None]:
    """
    Creates a GIF animation from sequentially numbered PNG images in the current directory.
    
    This function performs the following steps:
    1. Locates PNG image files starting with 'scene_' in the current directory.
    2. Extracts numerical values from the file names and sorts the images based on these values.
    3. Generates a GIF file named 'movement_{number}.gif' from the sorted images.
    4. Cleans up by removing the individual image files used to create the GIF.

    If no qualifying image files are found, it displays a message and exits.
    """
    # Current directory
    current_directory = os.getcwd()
    
    # Find PNG image files starting with 'scene_' in the current directory
    image_files = [file for file in os.listdir(current_directory) if file.endswith('.png') and file.startswith('scene_')]

    if not image_files:
        print("No image files found.")
        return

    # Extract numerical values from image names and sort the files based on these values
    image_numbers = [float(re.findall(r'scene_([\d.]+)\.png', file)[0]) for file in image_files]
    sorted_indices = sorted(range(len(image_numbers)), key=lambda k: image_numbers[k])
    sorted_image_files = [image_files[i] for i in sorted_indices]

    # Determine the filename for the GIF
    gif_int = 1
    gif_filename = f"movement_{gif_int}.gif"

    while os.path.exists(gif_filename):
        gif_int += 1
        gif_filename = f"movement_{gif_int}.gif"

    # Collect paths for the sorted image files
    images = [os.path.join(current_directory, img) for img in sorted_image_files]

    # Create the GIF using imageio
    with imageio.get_writer(gif_filename, loop=800, duration=1000) as writer:
        for image in images:
            frame = imageio.imread(image)
            writer.append_data(frame)

    # Remove the individual image files used for the GIF creation
    for img in sorted_image_files:
        os.remove(os.path.join(current_directory, img))
    return None
#%%

def remove_scene_images() -> Optional[None]:
    """
    Removes PNG image files starting with 'scene_' in the current directory.
    
    This function performs the following steps:
    1. Locates PNG image files starting with 'scene_' in the current directory.
    2. Removes these image files from the directory.

    If no qualifying image files are found, it exits gracefully without any action.
    """
    current_directory = os.getcwd()
    
    image_files = [file for file in os.listdir(current_directory) if file.endswith('.png') and file.startswith('scene_')]

    for img in image_files:
        os.remove(os.path.join(current_directory, img))
    
    return None
#%%

def create_gif_from_channel(images: List[str]) -> Optional[int]:
    """
    Creates a GIF animation from a list of image paths.

    Arguments:
    - images: A list of file paths for individual frames/images.

    This function performs the following steps:
    1. Determines the filename for the GIF based on existing files in the directory.
    2. Combines the provided images into a GIF file.
    3. Removes the individual image files used for the GIF creation.
    4. Returns the number of the generated GIF.

    If a similar filename already exists, it increments the number until finding a unique filename.
    """
    gif_int = 1
    gif_filename = f"channel_{gif_int}.gif"

    while os.path.exists(gif_filename):
        gif_int += 1
        gif_filename = f"channel_{gif_int}.gif"

    # Combine images into a GIF
    with imageio.get_writer(gif_filename, loop=800, duration=1000) as writer:
        for image in images:
            frame = imageio.imread(image)
            writer.append_data(frame)

    # Remove the temporary image files
    for image in images:
        os.remove(image)
    
    return gif_int

#%%

def cartesian_to_cylindrical(coordinates: List[float]) -> Optional[np.ndarray]:
    """
    Converts Cartesian coordinates to cylindrical coordinates.

    Arguments:
    - coordinates: List of three floats [x, y, z] representing Cartesian coordinates.

    Returns:
    - NumPy array representing cylindrical coordinates [r, theta, z_cylindrical].

    This function performs the following conversions:
    1. Calculates the radial distance 'r' from the origin.
    2. Determines the angle 'theta' in the xy-plane using arctan2.
    3. Preserves the z-coordinate as 'z_cylindrical'.
    """
    x, y, z = coordinates

    r = np.sqrt(x**2 + y**2)  
    theta = np.arctan2(y, x)  
    z_cylindrical = z  

    return np.array([r, theta, z_cylindrical])

#%%


def cylindrical_distance(point1: Tuple[float, float, float], point2: Tuple[float, float, float]) -> Optional[float]:
    """
    Computes the Euclidean distance between two points in cylindrical coordinates.

    Arguments:
    - point1: Tuple of three floats (r1, theta1, z1) representing cylindrical coordinates of point1.
    - point2: Tuple of three floats (r2, theta2, z2) representing cylindrical coordinates of point2.

    Returns:
    - Float representing the Euclidean distance between the two points in cylindrical space.

    This function calculates the Euclidean distance between two points given in cylindrical coordinates.
    """
    r1, theta1, z1 = point1
    r2, theta2, z2 = point2
    
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)
    
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    
    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    return distance

#%%
def cylindrical_to_cartesian(coordinates: List[float]) -> Optional[np.ndarray]:
    """
    Converts cylindrical coordinates to Cartesian coordinates.

    Arguments:
    - coordinates: List of three floats [r, theta, z] representing cylindrical coordinates.

    Returns:
    - NumPy array representing Cartesian coordinates [x, y, z].

    This function performs the following conversions:
    1. Calculates the x-coordinate using r*cos(theta).
    2. Calculates the y-coordinate using r*sin(theta).
    3. Preserves the z-coordinate as is.
    """
    r, theta, z = coordinates

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return np.array([x, y, z])
#%%

def intersection_point(r: float) -> Optional[float]:
    """
    Calculates the angle at which a line intersects a circle in cylindrical coordinates.

    Arguments:
    - r: Float representing the radius of the circle.

    Returns:
    - Float representing the angle theta where the line y = -3 intersects the circle of radius r.

    This function determines the angle theta at which the line y = -3 intersects a circle
    centered at the origin with radius r in polar coordinates.
    """
    # Equation of the line y = -3
    y_cartesian = -3
    
    # Calculate theta where r * sin(theta) = y
    theta = np.arcsin(y_cartesian / r)
    
    return theta
#%%

def basic_moving_towards(scene, random_positions: bool = False,
                         robot_initial_position_radius: List[float] = [0, 0],
                         robot_final_position_radius: List[float] = [0, 0]) -> Optional[np.ndarray]:
    """
    Moves a robot towards an antenna in a scene based on different positions.

    Arguments:
    - scene: Scene object or relevant data structure.
    - random_positions: Boolean flag indicating whether to use random positions for the robot.
    - robot_initial_position_radius: List of two floats [min_radius, max_radius] for the initial robot position.
    - robot_final_position_radius: List of two floats [min_radius, max_radius] for the final robot position.

    Returns:
    - NumPy array representing the movement vector if the robot moves towards the antenna, otherwise None.

    This function generates initial and final positions for a robot, checks if it moves towards an antenna,
    and updates the robot's position in the scene accordingly. It returns the movement vector if successful.
    """
    antenna_position = [0, 0, 0]
    antenna_position_cylindrical = cartesian_to_cylindrical(antenna_position)

    while True:

        if random_positions is False:

            initial_position = np.array([np.random.uniform(robot_initial_position_radius[0] ,robot_initial_position_radius[1])
                                        ,np.random.uniform(-np.pi/2, np.pi/2)
                                        ,0])


            final_position = np.array([np.random.uniform(robot_final_position_radius[0] ,robot_final_position_radius[1])
                                        ,np.random.uniform(-np.pi/2, np.pi/2)
                                        ,0])                                

        else:
                
            initial_position = np.array([np.random.uniform(0 ,2)
                                        ,np.random.uniform(-np.pi/2, np.pi/2)
                                        ,0])

            final_position = np.array([np.random.uniform(0, initial_position[0] )
                                        ,np.random.uniform(-np.pi/2, np.pi/2)
                                        ,0])                               
            


        initial_distance = cylindrical_distance(antenna_position_cylindrical, initial_position) # Distance between Robot and antenna before movement
        final_distance = cylindrical_distance(antenna_position_cylindrical, final_position) # Distance between Robot and antenna after movement

        print(f"Initial Position in cylindrical coordinate: {initial_position}")
        print(f"Initial Position in cartesian coordinate: {cylindrical_to_cartesian(initial_position)}")
        print(f"Final Position in cylindrical coordinate: {final_position}")
        print(f"Final Position in cartesian coordinate: {cylindrical_to_cartesian(final_position)}")

        if final_distance < initial_distance:
            print("Robot is moving towards the antenna.")
            # Changing the initial position of the Robot from the defult position of the Blender which is [0, 0, 0.94]
            params = mi.traverse(scene._scene)
            positions = dr.unravel(mi.Point3f, params['mesh-Robot.vertex_positions'])

            initial_position_cartesian = cylindrical_to_cartesian(initial_position)
            final_position_cartesian = cylindrical_to_cartesian(final_position)
            positions.x += initial_position_cartesian[0]
            positions.y += initial_position_cartesian[1]
            positions.z += initial_position_cartesian[2]

            params['mesh-Robot.vertex_positions'] = dr.ravel(positions)
            params.update()

            movement_vector = final_position_cartesian - initial_position_cartesian

            return movement_vector
        
        else:
            print('Robot does not move towards the antenna. Re-running the function.\n')


#%%

def moving_towards(scene, random_positions: bool = False,
                   robot_initial_position_radius: List[float] = [0, 0],
                   robot_final_position_radius: List[float] = [0, 0]) -> Optional[np.ndarray]:
    """
    Moves a robot towards an antenna in a scene based on different positions.

    Arguments:
    - scene: Scene object or relevant data structure.
    - random_positions: Boolean flag indicating whether to use random positions for the robot.
    - robot_initial_position_radius: List of two floats [min_radius, max_radius] for the initial robot position.
    - robot_final_position_radius: List of two floats [min_radius, max_radius] for the final robot position.

    Returns:
    - NumPy array representing the movement vector if the robot moves towards the antenna, otherwise None.

    This function generates initial and final positions for a robot, checks if it moves towards an antenna,
    and updates the robot's position in the scene accordingly. It returns the movement vector if successful.
    """

    antenna_position = [0, 0, 0]
    antenna_position_cylindrical = cartesian_to_cylindrical(antenna_position)

    while True:

        if random_positions is False:
            r_initial = np.random.uniform(robot_initial_position_radius[0], robot_initial_position_radius[1])
            initial_theta = intersection_point(r_initial)

            if np.isnan(initial_theta):
                initial_theta = np.random.uniform(-np.pi / 2, np.pi / 2)
            else:
                initial_theta = np.random.uniform(initial_theta / 1.15, -initial_theta / 1.15)

            initial_position = np.array([r_initial, initial_theta, 0])

            r_final = np.random.uniform(robot_final_position_radius[0], robot_final_position_radius[1])
            final_theta = intersection_point(r_final)

            if np.isnan(final_theta):
                final_theta = np.random.uniform(-np.pi / 2, np.pi / 2)
            else:
                final_theta = np.random.uniform(final_theta / 1.15, -final_theta / 1.15)

            final_position = np.array([r_final, final_theta, 0])

        else:
            initial_position_r = np.random.uniform(0, 2)
            initial_theta = intersection_point(initial_position_r)

            if np.isnan(initial_theta):
                initial_theta = np.random.uniform(-np.pi / 2, np.pi / 2)
            else:
                initial_theta = np.random.uniform(initial_theta / 1.15, -initial_theta / 1.15)

            initial_position = np.array([initial_position_r, initial_theta, 0])

            final_position_r = np.random.uniform(0, initial_position[0])
            final_theta = intersection_point(final_position_r)

            if np.isnan(final_theta):
                final_theta = np.random.uniform(-np.pi / 2, np.pi / 2)
            else:
                final_theta = np.random.uniform(final_theta / 1.15, -final_theta / 1.15)

            final_position = np.array([final_position_r, final_theta, 0])


        initial_distance = cylindrical_distance(antenna_position_cylindrical, initial_position) # Distance between Robot and antenna before movement
        final_distance = cylindrical_distance(antenna_position_cylindrical, final_position) # Distance between Robot and antenna after movement
        initial_position_cartesian = cylindrical_to_cartesian(initial_position)
        final_position_cartesian = cylindrical_to_cartesian(final_position)
        movement_vector = final_position_cartesian - initial_position_cartesian
        

        print(f"Initial Position in cylindrical coordinate: {initial_position}")
        print(f"Final Position in cylindrical coordinate: {final_position}")
        print(f"Initial Position in cartesian coordinate: {initial_position_cartesian}")
        print(f"Final Position in cartesian coordinate: {final_position_cartesian}\n")
        print(f"Movement vector: {movement_vector}")

        if final_distance < initial_distance:


            print("Robot is moving towards the antenna.\n")
            # Changing the initial position of the Robot from the defult position of the Blender which is [0, 0, 0.94]
            params = mi.traverse(scene._scene)
            positions = dr.unravel(mi.Point3f, params['mesh-Robot.vertex_positions'])
            #print(f" Position before: {positions}")

            positions.x += initial_position_cartesian[0]
            positions.y += initial_position_cartesian[1]
            positions.z += initial_position_cartesian[2]

            #print(f" Position after: {positions}\n")
            params['mesh-Robot.vertex_positions'] = dr.ravel(positions)
            params.update()

            return movement_vector
        
        else:
            print('Robot does not move towards the antenna. Re-running the function.\n')
#%%
# maybe should be removed
def interpolate_complex_series(series):

    real_part = series.apply(lambda x: x.real)
    imaginary_part = series.apply(lambda x: x.imag)

    # A DataFrame to perform interpolation
    df = pd.DataFrame({'Real': real_part, 'Imaginary': imaginary_part})
    df['Imaginary'] = df['Imaginary'].where(~np.isnan(df['Real']), None)

    df.index = series.index
  
    # Resample and interpolate real and imaginary parts
    resampled = df.resample('ns').interpolate(method='linear')

    # Reconstruct complex numbers
    interpolated_complex = resampled['Real'] + 1j * resampled['Imaginary']

    # New Series with the interpolated complex numbers
    interpolated_series = pd.Series(interpolated_complex, index=resampled.index, name = series.name)

    return interpolated_series

#%%

def abs_complex_columns(df):
    abs_df = pd.DataFrame()
    
    for col in df.columns:
        # Apply abs() to each element in the column (using complex)
        abs_values = df[col].apply(lambda x: abs(complex(x)))
        abs_df[col] = abs_values
    
    return abs_df

#%%
# maybe should be removed
def fill_missing_indexes(series):
    # Assuming the series index is a DatetimeIndex
    min_time = series.index.min()
    max_time = series.index.max()
    full_index = pd.timedelta_range(start=min_time, end=max_time, freq='ns')
    
    # Reindex the series with the full index and fill missing values with NaN
    full_series = series.reindex(full_index, fill_value=np.nan)
    
    missing_indexes = full_series.index.difference(series.index)
    
    return full_series, missing_indexes

#%%

def robot_positions_solver(random_positions: bool = False,
                             robot_initial_position_radius: List[float] = [0, 0],
                             robot_final_position_radius: List[float] = [0, 0]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates initial and final positions for a robot.

    Arguments:
    - random_positions: Boolean flag indicating whether to use random positions for the robot.
    - robot_initial_position_radius: List of two floats [min_radius, max_radius] for the initial robot position.
    - robot_final_position_radius: List of two floats [min_radius, max_radius] for the final robot position.

    Returns:
    - Tuple containing initial position, final position, initial distance to antenna, and final distance to antenna.
    """
    antenna_position = [0, 0, 0]
    antenna_position_cylindrical = cartesian_to_cylindrical(antenna_position)

    while True:

        if random_positions is False:
            r_initial = np.random.uniform(robot_initial_position_radius[0], robot_initial_position_radius[1])
            initial_theta = intersection_point(r_initial)

            if np.isnan(initial_theta):
                initial_theta = np.random.uniform(-np.pi / 2, np.pi / 2)
            else:
                initial_theta = np.random.uniform(initial_theta / 1.15, -initial_theta / 1.15)

            initial_position = np.array([r_initial, initial_theta, 0])

            r_final = np.random.uniform(robot_final_position_radius[0], robot_final_position_radius[1])
            final_theta = intersection_point(r_final)

            if np.isnan(final_theta):
                final_theta = np.random.uniform(-np.pi / 2, np.pi / 2)
            else:
                final_theta = np.random.uniform(final_theta / 1.15, -final_theta / 1.15)

            final_position = np.array([r_final, final_theta, 0])

        else:
            initial_position_r = np.random.uniform(0, 2)
            initial_theta = intersection_point(initial_position_r)

            if np.isnan(initial_theta):
                initial_theta = np.random.uniform(-np.pi / 2, np.pi / 2)
            else:
                initial_theta = np.random.uniform(initial_theta / 1.15, -initial_theta / 1.15)

            initial_position = np.array([initial_position_r, initial_theta, 0])

            final_position_r = np.random.uniform(0, initial_position[0])
            final_theta = intersection_point(final_position_r)

            if np.isnan(final_theta):
                final_theta = np.random.uniform(-np.pi / 2, np.pi / 2)
            else:
                final_theta = np.random.uniform(final_theta / 1.15, -final_theta / 1.15)

            final_position = np.array([final_position_r, final_theta, 0])


        initial_distance = cylindrical_distance(antenna_position_cylindrical, initial_position) # Distance between Robot and antenna before movement
        final_distance = cylindrical_distance(antenna_position_cylindrical, final_position) # Distance between Robot and antenna after movement
        initial_position_cartesian = cylindrical_to_cartesian(initial_position)
        final_position_cartesian = cylindrical_to_cartesian(final_position)
        movement_vector = final_position_cartesian - initial_position_cartesian
        
        if final_distance < initial_distance:

            print(f"Initial Position in cylindrical coordinate: {initial_position}")
            print(f"Final Position in cylindrical coordinate: {final_position}")
            print(f"Initial Position in cartesian coordinate: {initial_position_cartesian}")
            print(f"Final Position in cartesian coordinate: {final_position_cartesian}\n")

            return initial_position_cartesian , final_position_cartesian

#%%

def initial_position_prepration(scene, initial_position_cartesian: np.ndarray, final_position_cartesian: np.ndarray) -> Optional[np.ndarray]:
    """
    Moves the robot from the origin of the Blender ([0,0,0]) to its initial position.

    Arguments:
    - scene: Scene object or relevant data structure.
    - initial_position_cartesian: NumPy array representing the initial position in Cartesian coordinates.
    - final_position_cartesian: NumPy array representing the final position in Cartesian coordinates.

    Returns:
    - NumPy array representing the final movement vector.
    """

    movement_vector = final_position_cartesian - initial_position_cartesian

    # Changing the initial position of the Robot from the defult position of the Blender which is [0, 0, 0.94]
    params = mi.traverse(scene._scene)
    positions = dr.unravel(mi.Point3f, params['mesh-Robot.vertex_positions'])
    #print(f" Position before: {positions}")

    positions.x += initial_position_cartesian[0]
    positions.y += initial_position_cartesian[1]
    positions.z += initial_position_cartesian[2]

    #print(f" Position after: {positions}\n")
    params['mesh-Robot.vertex_positions'] = dr.ravel(positions)
    params.update()

    return movement_vector

#%%

def robot_rotation( initial_position_cartesian, final_position_cartesian):

    # Calculate the vector pointing from point A to point B
    movement_vector = final_position_cartesian - initial_position_cartesian

    # Calculate the normalized direction vector
    direction = movement_vector / np.linalg.norm(movement_vector)

    # Calculate the rotation angle around the Z-axis
    rotation_angle = np.arctan2(direction[1], direction[0])

    mesh = trimesh.load('C:\\Users\\dehghani\\Documents\\test_scene\\Robot.ply')

    # Convert angle to degrees and create rotation matrix
    rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, [0, 0, 1])

    # Apply rotation to the mesh
    mesh.apply_transform(rotation_matrix)

    mesh.export('C:\\Users\\dehghani\\Documents\\test_scene\\meshes\\Robot.ply', file_type='ply')

    return None
#%%