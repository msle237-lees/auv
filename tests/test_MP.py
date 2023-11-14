import numpy as np
from typing import Optional
import time
import pandas as pd
import matplotlib.pyplot as plt
import datetime

class PID:
    """
    A very basic PID controller with Integral clamping.
    """ 
    def __init__(self, Kp : float, Ki : float, Kd : float, i_max : float = None, i_min : float = None, output_max : float = None, output_min : float = None):
        """
        Initialize the PID controller with the given gains and clamping values.

        @param Kp Proportional gain.
        @param Ki Integral gain.
        @param Kd Derivative gain.
        @param i_max Maximum integral term.
        @param i_min Minimum integral term.
        @param output_max Maximum output value.
        @param output_min Minimum output value.
        """
        self.Kp = Kp 
        self.Ki = Ki 
        self.Kd = Kd

        self.i_max = i_max
        self.i_min = i_min
        self.output_max = output_max
        self.output_min = output_min
    
        self.set_point = 0.0
        self.error_sum = 0.0
        self.last_error = 0.0
        self.last_time = None

    def set_target(self, target : float = 0.0) -> float:
        """
        Set the desired target point for the controller.

        @param target Desired target value.
        """
        self.set_point = target

    def update(self, current_value : float, current_time : float) -> float:
        """
        Update the PID controller with the current value and time.

        @param current_value: The current value to control.
        @param current_time: The current time, as a floating-point timestamp.

        @return: The control variable.
        """
        error = self.set_point - current_value

        # Proportional term
        p_term = self.Kp * error

        # Integral term
        # This is computed only if a certain amount of time has passed
        if self.last_time is not None:
            time_diff = current_time - self.last_time
            self.error_sum += error * time_diff
            # Clamp the integral term if necessary
            if self.i_max is not None and self.error_sum > self.i_max:
                self.error_sum = self.i_max
            if self.i_min is not None and self.error_sum < self.i_min:
                self.error_sum = self.i_min
        else:
            time_diff = 0
        i_term = self.Ki * self.error_sum

        # Derivative term
        # This is computed only if a certain amount of time has passed
        d_term = 0
        if time_diff > 0:
            d_term = self.Kd * (error - self.last_error) / time_diff

        # Remember last error and last time for next update
        self.last_error = error
        self.last_time = current_time

        # Compute the control variable
        control_variable = p_term + i_term + d_term

        # Clamp the output if necessary
        if self.output_max is not None and control_variable > self.output_max:
            control_variable = self.output_max
        if self.output_min is not None and control_variable < self.output_min:
            control_variable = self.output_min

        return control_variable
    
class Movement_Package:
    """
    \breif Movement Package (MP) class for the AUV.
    Handles the creation and manipulation of thruster matrices for either
    real-world or simulation scenarios. Three different configurations have been
    created for the AUV, which are determined by the placement parameter. 
    """
    def __init__(self, simulation : bool = False, num_dof : int = 6, num_motors: int = 8):
        """
        \breif Initialize the Movement_Package (MP) class with given parameters.
        \param placement: Determines what motor placement should be active.
                            Currently three different configurations have been created.
        \param simulation: Determines if the thruster matrix should be set up
                            for a real-world or simulation environment.
        \param num_dof: Number of degrees of freedom for the AUV.
        \param num_motors: Number of motors for the AUV.
        """
        self.simulation = simulation
        self.num_dof = num_dof
        self.num_thrusters = num_motors
        self.sensor_data = np.zeros(self.num_dof)

        self.horizontal_motor_angles = 45.0
        self.horizontal_motor_distance = 0.22
        self.vertical_motor_distance = 0.22

        # Convert angle from degrees to radians
        self.angle_in_radians = self.horizontal_motor_angles * (np.pi / 180)

        self.thruster_matrix = self.create_thruster_matrix().T

        # PID controller parameters, these are set to zero for now. Expected to be tuned later.
        # absolute value Kp values should range from 0.1 to 1.0
        # absolute value Ki values should range from 0.0 to 0.5
        # absolute value Kd values should range from 0.0 to 1.0
        Kp_List = [0.7, 0.6, 0.6, 0.6, 0.6, 0.6]
        Ki_List = [0.25, 0.05, 0.05, 0.05, 0.05, 0.05]
        Kd_List = [0.0, 0.05, 0.05, 0.05, 0.05, 0.05]
        i_max_List = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        i_min_List = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        output_max_List = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        output_min_List = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        self.PIDs = self.init_PID(Kp_List, Ki_List, Kd_List, i_max_List, i_min_List, output_max_List, output_min_List)

    def init_PID(self, Kp_List : list, Ki_List : list, Kd_List : list, i_max_List : list, i_min_List : list, output_max_List : list, output_min_List : list) -> list:
        """
        \breif Initialize the PID controller with the given gains and clamping values.

        \param Kp Proportional gain.
        \param Ki Integral gain.
        \param Kd Derivative gain.
        \param i_max Maximum integral term.
        \param i_min Minimum integral term.
        \param output_max Maximum output value.
        \param output_min Minimum output value.
        """
        PIDs = []
        for i in range(self.num_dof):
            PIDs.append(PID(Kp_List[i], Ki_List[i], Kd_List[i], i_max_List[i], i_min_List[i], output_max_List[i], output_min_List[i]))
        return PIDs

    def create_thruster_matrix(self, simulation: bool = False) -> np.ndarray:
        """
        Create the thruster matrix for either a real-world or simulated scenario.

        @param simulation bool: If True, creates the matrix for simulation. Otherwise, for the real world.
        @return np.ndarray: The thruster matrix for the specified scenario.
        """
        thruster_matrix = np.zeros((self.num_thrusters, self.num_dof))
        
        # Compute forces and moments for horizontal thrusters
        for i in range(4):
            # Calculate the angle of the thruster
            angle = np.deg2rad(self.horizontal_motor_angles + 90 * i)

            # Force direction
            d = np.array([np.cos(angle), np.sin(angle), 0])
            
            # Moment calculation - cross product of position vector and force direction
            position = np.array([self.horizontal_motor_distance * np.cos(angle), 
                                self.horizontal_motor_distance * np.sin(angle), 
                                0])
            m = np.cross(position, d)
            
            thruster_matrix[i, :3] = d
            thruster_matrix[i, 3:] = m

        # Configure vertical thrusters for Z-axis, Roll, and Pitch
        vertical_thruster_sign = -1 if simulation else 1
        for i in range(4, 8):
            # Force direction - pointing downwards
            d = np.array([0, 0, vertical_thruster_sign])

            # Moment calculation - cross product of position vector and force direction
            position = np.array([self.vertical_motor_distance * np.cos(np.deg2rad(90 * (i - 4))),
                                self.vertical_motor_distance * np.sin(np.deg2rad(90 * (i - 4))),
                                0])
            m = np.cross(position, d)
            
            thruster_matrix[i, :3] = d
            thruster_matrix[i, 3:] = m

        return thruster_matrix

    def update(self, desired_data: np.ndarray, sensor_data: np.ndarray) -> np.ndarray:
        """
        Update the input data for the AUV and calculate the returned PID values. 
        Apply the map_data function to the returned values and return the mapped values.
        @param desired_data: The desired data for the AUV.
        @param sensor_data: The sensor data for the AUV.
        @return: The returned PID values mapped to the motor values (1x8 matrix).
        """
        pid_output = np.zeros(self.num_dof)
        for i in range(len(self.PIDs)):
            self.PIDs[i].set_target(desired_data[i])
            pid_output[i] = self.PIDs[i].update(sensor_data[i], time.time())

        # Multiply the PID output with the thruster matrix
        # Uses the PID output as the desired data (Useful for testing the PID controller)
        thruster_output = pid_output.dot(self.thruster_matrix)
        # Uses the desired data as the desired data (Useful for testing the thruster matrix)
        # thruster_output = desired_data.dot(self.thruster_matrix)

        # Map the entire thruster output using the map_data function
        mapped_values = self.map_data(thruster_output)

        # Run the sensor update function to get the sensor data
        sensor_data = self.sensor_update(mapped_values, 0.1)

        return pid_output, mapped_values

    def sensor_update(self, thruster_output: np.ndarray, time_step: float) -> np.ndarray:
        # Read thrust data from the CSV file and ensure numeric conversion
        thrust_data = pd.read_csv('motor_data/T200.csv', header=None)
        thrust_data[0] = pd.to_numeric(thrust_data[0], errors='coerce')
        thrust_data[1] = pd.to_numeric(thrust_data[1], errors='coerce')
        thrust_data = thrust_data.dropna().to_numpy()

        # Constants and placeholders
        drag_matrix = np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])  # Placeholder drag coefficients
        auv_mass = 20.0  # Mass of the AUV in kg
        auv_inertia = np.array([10.0, 10.0, 10.0])  # Placeholder inertia

        # Initialize arrays for forces and torques
        total_force = np.zeros(3)  # Surge, Sway, Heave
        total_torque = np.zeros(3)  # Roll, Pitch, Yaw

        # Update logic for thrusters
        for i in range(self.num_thrusters):
            # Find the closest PWM value in the thrust data and get the corresponding thrust value
            pwm_value = thruster_output[i]
            closest_pwm_index = np.abs(thrust_data[:, 0] - pwm_value).argmin()
            force = thrust_data[closest_pwm_index, 1]

            # Calculate angle and position for horizontal and vertical thrusters
            if i < 4:  # Horizontal thrusters
                angle = np.deg2rad(self.horizontal_motor_angles + 90 * i)
                direction = np.array([np.cos(angle), np.sin(angle), 0])
                position = np.array([self.horizontal_motor_distance * np.cos(angle), 
                                    self.horizontal_motor_distance * np.sin(angle), 
                                    0])
            else:  # Vertical thrusters
                direction = np.array([0, 0, -1])  # Assuming downward force
                position = np.array([self.vertical_motor_distance * np.cos(np.deg2rad(90 * (i - 4))),
                                    self.vertical_motor_distance * np.sin(np.deg2rad(90 * (i - 4))),
                                    0])

            # Add the force and torque contributions of each thruster
            total_force += force * direction
            total_torque += np.cross(position, force * direction)

        # Convert total force and torque to acceleration
        linear_acc = total_force / auv_mass
        angular_acc = total_torque / auv_inertia

        # Update sensor data (velocity) based on total acceleration
        self.sensor_data[:3] += linear_acc * time_step  # Update linear velocity (surge, sway, heave)
        self.sensor_data[3:] += angular_acc * time_step  # Update angular velocity (roll, pitch, yaw)

        # Apply drag (assuming linear drag model)
        self.sensor_data -= drag_matrix * self.sensor_data * time_step

        return self.sensor_data

    def map_data(self, data):
        """
        Map the output data from the PID controller to actual motor values.
        @param data: The output data from the PID controller (1x8 matrix).
        @return: The mapped motor values (1x8 matrix).
        """
        in_min, in_max = -1, 1
        out_min, out_max = 1000, 2000

        out_data = np.zeros(data.shape)  # Initialize an array of the same shape as input

        # Map each element in the thruster data to the new range
        for i in range(len(data)):
            out_data[i] = round((data[i] - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

        return out_data

    def run(self, desired_values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), num_iterations=100, time_step=0.1, output_file = "thruster_outputs.csv"):
        """
        Run the movement package for testing and tuning the PID controllers.
        @param num_iterations: Number of iterations to run the simulation.
        @param time_step: Time step for each iteration in seconds.
        """
        # Prompt for initial desired and sensor values
        sensor_values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        pid_outputs = np.zeros((num_iterations, self.num_dof))
        thruster_outputs = np.zeros((num_iterations, self.num_thrusters))
        logged_sensor_values = np.zeros((num_iterations, self.num_dof))
        logged_desired_values = np.zeros((num_iterations, self.num_dof))

        for i in range(num_iterations):
            logged_desired_values[i] = desired_values
            logged_sensor_values[i] = sensor_values

            pid_output, thruster_output = self.update(desired_values, sensor_values)
            pid_outputs[i] = pid_output
            thruster_outputs[i] = thruster_output

            # Update sensor values using the sensor_update function
            sensor_values = self.sensor_update(thruster_output, time_step)

            time.sleep(time_step)

        # Create DataFrame from the logged data
        time_axis = np.arange(0, num_iterations * time_step, time_step)
        data = {'Time': time_axis}
        for i in range(self.num_dof):
            data[f'Desired_{i+1}'] = logged_desired_values[:, i]
            data[f'Sensor_{i+1}'] = logged_sensor_values[:, i]
            data[f'PID_{i+1}'] = pid_outputs[:, i]
        for i in range(self.num_thrusters):
            data[f'Thruster_{i+1}'] = thruster_outputs[:, i]
        df = pd.DataFrame(data)

        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Data logged to {output_file}")

        # Plotting the thruster outputs
        time_axis = np.arange(0, num_iterations * time_step, time_step)
        
        # Create a figure with subplots arranged in 3 rows and 3 columns
        fig, axs = plt.subplots(3, 3, figsize=(15, 10))  # Adjust figsize as needed

        for i in range(self.num_thrusters):
            # Determine the row and column to place each subplot
            row = i // 3
            col = i % 3
            axs[row, col].plot(time_axis, thruster_outputs[:, i], label=f'Thruster {i+1}')
            axs[row, col].set_xlabel('Time (seconds)')
            axs[row, col].set_ylabel('Output')
            axs[row, col].set_title(f'Thruster {i+1} Output Over Time')
            axs[row, col].legend()

        # Hide the last subplot (which is empty)
        axs[-1, -1].axis('off')

        plt.tight_layout()  # Adjusts the plots to fit into the figure area.
        plt.show()

if __name__ == "__main__":
    mp = Movement_Package()
                    #   np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0]),\
                    #   np.array([0.0, -0.5, 0.0, 0.0, 0.0, 0.0]),\
                    #   np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),\
                    #   np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0]),\
                    #   np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),\
                    #   np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0]),\
                    #   np.array([0.0, 0.0, -0.5, 0.0, 0.0, 0.0]),\
                    #   np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),\
                    #   np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),\
                    #   np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),\
                    #   np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0]),\
                    #   np.array([0.0, 0.0, 0.0, -0.5, 0.0, 0.0]),\
                    #   np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),\
                    #   np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0]),\
                    #   np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),\
                    #   np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0]),\
                    #   np.array([0.0, 0.0, 0.0, 0.0, -0.5, 0.0]),\
                    #   np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),\
                    #   np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0]),\
                    #   np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),\
                    #   np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),\
                    #   np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.5]),\
                    #   np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),\
                    #   np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),\
                    #   np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),\
    desired_values = [np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),\
                      np.array([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),\
                      np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),\
                      np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),\
                      np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),\
                      ]
    for desired_value in desired_values:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        mp.run(desired_values=desired_value, output_file=f"data_out/{timestamp}_thruster_outputs.csv")