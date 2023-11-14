import numpy as np
from typing import Optional
import time

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
        p_term = self.p_gain * error

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
        i_term = self.i_gain * self.error_sum

        # Derivative term
        # This is computed only if a certain amount of time has passed
        d_term = 0
        if time_diff > 0:
            d_term = self.d_gain * (error - self.last_error) / time_diff

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
    \brief Initialize the Movement_Package (MP) class with given parameters.
    
    Handles the creation and manipulation of thruster matrices for either
    real-world or simulation scenarios.

    \param placement: Determines what motor placement should be active. 
                      Currently three different configurations have been created.
    \param simulation: Determines if the thruster matrix should be set up
                       for a real-world or simulation environment.
    """
    
    def __init__(self, placement : int = 0, simulation : bool = False, num_dof : int = 6):
        """ \brief Constructor for initializing the Movement_Package class.

        \param placement (int): Motor configuration choice, options are 1-3 and determine where motors are placed.
        \param simulation (bool): Determines if thruster matrix is for real-world or simulation.
        \param num_dof (int): Controller input dictates the number of DOF available to the machine. default is 6.
        """
        # Total number of thrusters
        self.num_thrusters = 8  
        # Degrees of freedom (X, Y, Z, Roll, Pitch, Yaw)
        self.num_dof = num_dof
        
        # Thruster angle in degrees
        self.angle_in_degrees = 45  
        # Distance from center of mass to thrusters
        self.distance_from_center = 1.0
        
        # Convert angle from degrees to radians
        self.angle_in_radians = self.angle_in_degrees * (np.pi / 180)
        
        # Initialize thruster data array
        self.thruster_data = np.zeros(self.num_thrusters)
        
        # Create thruster matrix based on simulation flag
        if not simulation:
            self.thruster_matrix = self.create_thruster_matrix_real_world()
        else:
            self.thruster_matrix = self.create_thruster_matrix_simulation()

        # Create the PID Matrix and initialize it as a column vector of size num_dof x 1 
        for i in range(0, num_dof):
            i = PID(Kp = 0.1, Ki = 0.1, Kd = 0.1, )

    def create_thruster_matrix_real_world(self):
        """ \brief Create the thruster matrix for a real-world scenario.
        
        \return np.ndarray: The real-world thruster matrix.
        """
        thruster_matrix = np.zeros((self.num_thrusters, self.num_dof))
        
        # Compute forces and moments for horizontal thrusters
        for i in range(4):
            angle_adjustment = np.pi if i % 2 == 1 else 0
            d = np.array([np.cos(self.angle_in_radians + angle_adjustment), np.sin(self.angle_in_radians + angle_adjustment), 0])
            m = np.array([0, 0, self.distance_from_center * np.sin(self.angle_in_radians + angle_adjustment)])
            thruster_matrix[i, :3] = d
            thruster_matrix[i, 3:] = m

        # Configure vertical thrusters for Z-axis, Roll, and Pitch
        for i in range(4, 8):
            d = np.array([0, 0, 1])
            m = np.array([(-1)**((i - 4) // 2) * self.distance_from_center,  (-1)**((i - 4) % 2) * self.distance_from_center, 0])
            thruster_matrix[i, :3] = d
            thruster_matrix[i, 3:] = m

        return thruster_matrix

    def create_thruster_matrix_simulation(self):
        """ \brief Create the thruster matrix for a simulated scenario.
        
        \return np.ndarray: The simulated thruster matrix.
        """
        thruster_matrix = np.zeros((self.num_thrusters, self.num_dof))
        
        # Compute forces and moments for horizontal thrusters
        for i in range(4):
            angle_adjustment = np.pi if i % 2 == 1 else 0
            d = np.array([np.cos(self.angle_in_radians + angle_adjustment), np.sin(self.angle_in_radians + angle_adjustment), 0])
            m = np.array([0, 0, self.distance_from_center * np.sin(self.angle_in_radians + angle_adjustment)])
            thruster_matrix[i, :3] = d
            thruster_matrix[i, 3:] = m

        # Configure vertical thrusters for Z-axis, Roll, and Pitch
        for i in range(4, 8):
            d = np.array([0, 0, -1])
            m = np.array([(-1)**((i - 4) // 2) * self.distance_from_center,  (-1)**((i - 4) % 2) * self.distance_from_center, 0])
            thruster_matrix[i, :3] = d
            thruster_matrix[i, 3:] = m

        return thruster_matrix

    def update(self, data):
        """ \brief Update thruster outputs based on desired vehicle movements.
        
        \param data (list): List of desired movements in the 6 DOF.
        \return np.ndarray: Updated thruster data.
        """
        data = np.transpose(np.array(data))
        print(data)

    def map_data(self):
        """ \brief Map the thruster outputs to a new range suitable for actuation.
        
        \return np.ndarray: Mapped thruster data.
        """
        in_min, in_max = -1, 1  
        out_min, out_max = 1000, 2000  

        out_data = np.zeros(len(self.thruster_data))
        
        # Map the thruster data to the new range
        for x in range(len(self.thruster_data)):
            out_data[x] = (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min

        return out_data

    def print_out(self):
        """ \brief Print the current thruster outputs. """
        print(f'Thruster outputs: {self.thruster_data}')

    def run(self):
        """ \brief Main loop to run the program. Takes user input for desired movements. """
        while True:
            try:
                data = input("Enter movement data (comma separated, 6 values): ")
                data = data.split(',')
                data = [float(i) for i in data]
                self.update(data)
                self.print_out()
            except KeyboardInterrupt:
                print('Exiting...')
                break
if __name__ == "__main__":
    mp = Movement_Package(simulation=False)
    print(mp.thruster_matrix.T)
    mp.run()
