import numpy as np
import mitsuba as mi
import drjit as dr
import pandas as pd
from helper import *
from sionna.rt.solver_paths import SolverPaths as Solver


def simulate_robot_movement(scene, robot_speed: float = 1.6, time_step: float = 0.01,
            initial_position_cartesian: list = [0, 0], final_position_cartesian: list = [0, 0]) -> list:
    """
    Simulates robot movement through a scene, allowing for tracking channel characteristics over time.

    Parameters:
    - scene (sionna.rt.scene.Scene): Scene loaded by sionna.
    - robot_speed (float, optional): Speed of the robot in meters per second. Default is 1.6 m/s.
    - time_step (float, optional): Time step in seconds. Default is 0.01 seconds.
    - initial_position (numpy.ndarray or 'Random', optional): Represents the initial position of the robot.
        If 'Random', the initial position is randomly assigned within the scene. Default is 'Random'.
    - movement_vector (numpy.ndarray or 'Random', optional): A 3D vector representing the direction of movement.
        If 'Random', the movement vector is randomly chosen. Default is 'Random'.

    Returns:
    - channel_realization_list (list): A list containing channel characteristics data as pandas Series
        at different time steps during the robot's movement.
    """
    print(f"Time Step: {time_step} s")
    print(f"Robot Speed: {robot_speed} m/s \n")

    #movement_vector = moving_towards(scene = scene,  random_positions = random_positions,
    #               robot_initial_position_radius = robot_initial_position_radius,
    #               robot_final_position_radius = robot_final_position_radius)
    
    movement_vector = initial_position_prepration(scene, initial_position_cartesian, final_position_cartesian)



    delta_d = np.linalg.norm(movement_vector)
    normalized_movement_vector = movement_vector / delta_d

    # Calculate time and new positions
    delta_t = delta_d / robot_speed
    
    print(f"Total Distance to be Traveled: {round(delta_d,2)} meters")
    print(f"Total Time to Complete the Movement: {round(delta_t,2)} seconds")

    # Initialize channel_realization_list
    channel_realization_list = []

    # Iterate from 0 to delta_t in steps of time_step
    current_time = 0
    while current_time <= delta_t:
        current_time = round(current_time, 2)

        # Calculate new positions
        params = mi.traverse(scene._scene)
        positions = dr.unravel(mi.Point3f, params['mesh-Robot.vertex_positions'])

        if current_time == 0:
            positions = positions
        else:
            positions.x += (time_step * movement_vector[0])/ delta_t
            positions.y += (time_step * movement_vector[1])/ delta_t
            positions.z += (time_step * movement_vector[2])/ delta_t

        params['mesh-Robot.vertex_positions'] = dr.ravel(positions)
        params.update()

        scene._solver_paths = Solver(scene)
        paths = scene.compute_paths(method='fibonacci', max_depth=1, num_samples=1e7, reflection=False, scattering=True,
                                    diffraction=False, los=False)
        paths.normalize_delays = False
        a, tau = paths.cir(los=False, reflection=False, diffraction=False, scattering=True)

        # Render scene with new camera*
        #scene.render(camera="my_cam",show_devices=True)
        scene.render_to_file(camera="my_cam", filename= f"scene_{current_time}.png",resolution=[700,550] )
        

        a = a.numpy().squeeze()
        tau = tau.numpy().squeeze()

        time_series = pd.Series(a, index = tau , name = current_time).sort_index()
        channel_realization_list.append(time_series)

        current_time += time_step

    return channel_realization_list