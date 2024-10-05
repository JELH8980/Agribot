# -----------------------------------------------------------------------------
# Author: Ludwig Horvath
# Email: ludhor@kth.se
# Date: 2024-10-05
# -----------------------------------------------------------------------------


# Standard library imports
import os
import re
from datetime import datetime
import warnings
import time

# Numerical and data handling libraries
import pandas as pd
import numpy as np
import copy
from itertools import product
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
import joblib  # Import joblib for saving the model

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
import joblib  # Import joblib for saving the model


# Custom modules
from simulation import Sim
from settings import saved_features
from tqdm import tqdm

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Get current date and time for file naming
today = datetime.now().strftime('%Y-%m-%d')
time_of_day = datetime.now().strftime('%H-%M')

# Jumpscare toggle flag
Surprise = False

def print_simulation_parameters(param_dict):
    """
    Prints the parameters of the simulation in a structured format.

    Args:
        param_dict (dict): Dictionary containing simulation parameters.
    """
    print("Running simulation with parameters:")
    for key, value in param_dict.items():
        if isinstance(value, dict):  # Check if the value is a dictionary
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")


# ===================================
# Post-Simulation Analysis Classes
# ===================================


class Video:
    def __init__(self, data, speed=400, fps=30):
        """
        Initializes the Video class for animating simulation data.

        Args:
            data: Simulation data object that contains positions and states of agents.
            speed (int): Speed of the animation in milliseconds (default: 400ms).
            fps (int): Frames per second for the animation (default: 30).
        """
        self.fig, self.ax = None, None
        self.data = data  # Simulation data
        self.artists = []  # To store frames for animation
        self.features = {}  # Store static features like paths, rectangles, nodes
        self.x_bounds = [-1, 20]  # Bounds for x-axis in the 2D plot
        self.y_bounds = [-1, 27]  # Bounds for y-axis in the 2D plot

        # Load visual settings with defaults
        self.visual_settings = saved_features['plotstyle_settings']
        plt.rcParams['font.family'] = self.visual_settings.get('font_family', 'serif')
        plt.rcParams['font.serif'] = [self.visual_settings.get('font_name', 'Times New Roman')]
        

        # Robot dimensions fetched from saved features
        self.robot_width = saved_features['agents']['robot']['width']
        self.robot_height = saved_features['agents']['robot']['height']

        # Animation controls
        self.ani_running = True
        self.interval = speed  # Time interval between frames (controls animation speed)
        self.fps = fps  # Frames per second for animation
        self.animation = None  # Animation object placeholder
        self.stride = 10  # Frame stride, controls how often to capture frames
        self.current_frame = 0  # Track current frame for replaying animation

        # Deep copy of saved features to avoid mutating original
        self.saved_features_copy = copy.deepcopy(saved_features)

        # Optional jumpscare feature
        self.jumpscare = Surprise


    def init_figure_axis(self):
        """
        Initializes the figure and axis for the 2D environment animation.
        Sets axis limits and labels.
        """
        self.fig, self.ax = plt.subplots(figsize=(10, 10), num='2D Animation')
        self.ax.set_xlim(self.x_bounds)  # Set x-axis bounds
        self.ax.set_ylim(self.y_bounds)  # Set y-axis bounds
        self.ax.set_aspect('equal')  # Maintain equal aspect ratio
        self.ax.set_title('2D Environment')
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')

    def load_static_features(self):
        """
        Loads and adds static elements like paths, rectangles, and nodes to the plot.
        These elements don't change during the animation.
        """
        # Clear previous static features if replayed
        self.features.clear()

        # Add static patches (rectangles) for areas like ground and farms
        self.features['ground'] = self.ax.add_patch(self.saved_features_copy['rectangles']['ground'])
        self.features['farm_1'] = self.ax.add_patch(self.saved_features_copy['rectangles']['farm1'])
        self.features['farm_2'] = self.ax.add_patch(self.saved_features_copy['rectangles']['farm2'])
        self.features['farm_3'] = self.ax.add_patch(self.saved_features_copy['rectangles']['farm3'])

        # Add static lines (paths) for routes between locations
        self.features['path_c1'] = self.ax.add_line(self.saved_features_copy['paths']['charging_to_farm_1']['Line'])
        self.features['path_1'] = self.ax.add_line(self.saved_features_copy['paths']['farm_1']['Line'])
        self.features['path_12'] = self.ax.add_line(self.saved_features_copy['paths']['farm_transition_1']['Line'])
        self.features['path_2'] = self.ax.add_line(self.saved_features_copy['paths']['farm_2']['Line'])
        self.features['path_23'] = self.ax.add_line(self.saved_features_copy['paths']['farm_transition_2']['Line'])
        self.features['path_3'] = self.ax.add_line(self.saved_features_copy['paths']['farm_3']['Line'])
        self.features['path_3c'] = self.ax.add_line(self.saved_features_copy['paths']['farm_3_to_charging']['Line'])

        # Add checkpoints or nodes at various locations
        self.features['node_1'] = self.ax.scatter(**self.saved_features_copy['checkpoints']['entrance/exit'])
        self.features['node_2'] = self.ax.scatter(**self.saved_features_copy['checkpoints']['farm1'])
        self.features['node_3'] = self.ax.scatter(**self.saved_features_copy['checkpoints']['farm2'])
        self.features['node_4'] = self.ax.scatter(**self.saved_features_copy['checkpoints']['farm3'])
        self.features['node_5'] = self.ax.scatter(**self.saved_features_copy['checkpoints']['chargingstation'])

        # Display state labels with initial values
        self.features['State labels'] = self.fig.text(
            0.8, 0.75,  # Position in figure coordinates (outside axes)
            f"Time: \n\nM\nU\nD*\nD",  # Placeholder text
            ha='left', va='center', fontsize=12,
            bbox=dict(boxstyle="square,pad=0.5", edgecolor="white", facecolor="white")
        )

        # Add clarification image to the figure
        image_clarification = mpimg.imread('Assets/Images/SimulationClarification.png')  
        self.features['image_clarification'] = self.fig.figimage(image_clarification, xo=100, yo=400, alpha=1)  # Adjust position

    def add_artist(self):
        """
        Adds dynamic elements to the plot such as robots and humans.
        Creates frames for the animation.
        """
        self.artists.clear()  # Clear previous artists if replayed

        for time_indx, _ in enumerate(self.data['time']):
            if time_indx % self.stride == 0:  # Skip frames based on stride
                artist_list = []

                # Loop through all agents and add their positions to the frame
                for agent_key in self.data['position'][time_indx].keys():
                    if bool(re.fullmatch(r"(worker\d+|visitor\d+)", agent_key)):
                        # Worker agents (blue) and visitor agents (red)
                        position = self.data['position'][time_indx][agent_key][0]
                        color = 'blue' if 'worker' in agent_key else 'red'
                        symbol = self.ax.scatter(position[0], position[1], color=color, zorder=5)
                        artist_list.append(symbol)

                # Add robot and safety zones
                robot_position = self.data['position'][time_indx]['robot']
                robot_anchor_position = robot_position - np.array([self.robot_height/2, self.robot_width/2])

                green_zone = self.ax.add_patch(patches.Circle(xy=robot_position, **self.saved_features_copy['informative']['safety zones']['green zone']))
                yellow_zone = self.ax.add_patch(patches.Circle(xy=robot_position, **self.saved_features_copy['informative']['safety zones']['yellow zone']))
                red_zone = self.ax.add_patch(patches.Circle(xy=robot_position, **self.saved_features_copy['informative']['safety zones']['red zone']))
                robot_symbol = self.ax.add_patch(patches.Rectangle(xy=robot_anchor_position, **self.saved_features_copy['agents']['robot']))

                # Display state values (M, U, D*, D)
                M = self.data['state'][time_indx]['M']
                U = self.data['state'][time_indx]['U']
                D_star = self.data['state'][time_indx]['D*']
                D = self.data['state'][time_indx]['D']
                time = np.round(self.data['time'][time_indx], 2)

                state_values = self.fig.text(0.85, 0.75, f"{time}\n\n{int(M)}\n{int(U)}\n{D_star}\n{D}",
                                             ha='left', va='center', fontsize=12,
                                             bbox=dict(boxstyle="square,pad=0.5", edgecolor="white", facecolor="white"))

                # Append artists (robots, zones, humans, and state values) for animation
                artist_list.extend([robot_symbol, green_zone, yellow_zone, red_zone, state_values])
                self.artists.append(artist_list)

    def create_animation(self):
        """
        Creates the animation by combining the static and dynamic elements.
        Does not play the animation yet.
        """
        self.init_figure_axis()  # Initialize the figure and axis
        self.load_static_features()  # Load static elements (like paths, nodes)
        self.add_artist()  # Add dynamic elements (robot, humans, etc.)
        self.animation = animation.ArtistAnimation(self.fig, self.artists, interval=self.interval)  # Create the animation

    def play_animation(self):
        """
        Plays the created animation.
        If jumpscare is enabled, it triggers after the animation starts.
        """
        if self.animation is not None:
            plt.show()  # Show the animation plot
            if self.jumpscare:
                self.jump_scare()  # Trigger jumpscare if set
        else:
            print("No animation available to play. Please create it first.")

    def toggle_animation(self):
        """
        Toggles the animation between start and stop.
        """
        if self.ani_running:
            self.animation.event_source.stop()
        else:
            self.animation.event_source.start()
        self.ani_running = not self.ani_running

    def save_animation(self, case):
        """
        Saves the animation as a GIF file with a timestamped directory.

        Args:
            case (str): A string to name the case or scenario for the animation.
        """
        base_directory = 'Results'
        os.makedirs(base_directory, exist_ok=True)  # Ensure base directory exists

        # Create timestamped folder for saving animation
        folder_path = os.path.join(base_directory, today)
        time_folder_path = os.path.join(folder_path, time_of_day)
        os.makedirs(time_folder_path, exist_ok=True)  # Ensure time folder exists

        # Save the animation as a GIF
        new_file_path = os.path.join(time_folder_path, f'P_{time_of_day}_{case}.gif')
        self.animation.save(new_file_path, writer='pillow', fps=self.fps)

    def jump_scare(self):
        """
        A jumpscare feature that plays a sound and displays an image when triggered.
        """
        import pygame
        pygame.mixer.init()

        # Play sound effect
        sound_effect = pygame.mixer.Sound("Assets/Sounds/Sound1.mp3")
        sound_effect.play()

        # Disable UI toolbar for full immersion
        plt.rcParams['toolbar'] = 'None'

        # Create a full-screen black figure for jumpscare
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        # Load and display jumpscare image
        img = mpimg.imread('Assets/Images/JumpscareImage.jpg')
        ax.imshow(img, aspect='auto')
        ax.axis('off')  # Hide axes

        # Maximize figure to full screen
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()

        plt.show()  # Show jumpscare image



class Sensitivity_analysis:
    def __init__(self):
        """
        Initialize the Sensitivity_analysis class.
        Sets up file paths, loads existing performance data, and retrieves saved feature configurations.
        """
        self.metadata = None  # Placeholder for simulation metadata
        self.filepath = os.path.join('Results', 'metadata.csv')  # Path to save/load performance data
        self.performance_data = self.load_existing_data()  # Load existing performance data from CSV, if available
        self.saved_features = saved_features  # Reference to saved features for sensitivity analysis

    def load_existing_data(self):
        """
        Load existing data from the 'metadata.csv' file if it exists.
        Returns:
            pd.DataFrame: Loaded data or an empty DataFrame if file not found.
        """
        if os.path.exists(self.filepath):
            try:
                # Attempt to load CSV into a pandas DataFrame
                return pd.read_csv(self.filepath)
            except Exception as e:
                print(f"Error loading existing data: {e}")
                return pd.DataFrame()  # Return empty DataFrame if there's an error
        else:
            # If the file doesn't exist, return an empty DataFrame
            return pd.DataFrame()

    def calc_performance_metrics(self, data):
        """
        Calculate performance metrics from the given data.
        
        Args:
            data (pd.DataFrame): DataFrame containing 'Iteration', 'Outcome', and 'Duration' columns.

        Returns:
            dict: Dictionary containing calculated performance metrics.
        """

        # Calculate total unique iterations
        total_iterations = data['Iteration'].nunique()
        
        # Calculate the number of injury cases where the outcome is 'I'
        num_injuries = (data['Outcome'] == 'I').sum()
        
        # Compute the risk of injury as a ratio of injury cases to total iterations
        risk_of_injury = num_injuries / total_iterations if total_iterations > 0 else 0
        
        # Package the performance metrics into a dictionary
        performance_metrics = {
            'Total Iterations': total_iterations,
            'Number of Injury Cases': num_injuries,
            'Risk of Injury': risk_of_injury,
        }

        return performance_metrics

    def store_performance_metrics(self, sensor_parameters, nr_workers, nr_visitors, performance_metrics):
        """
        Store the calculated performance metrics into a CSV file.

        Args:
            sensor_parameters (dict): Dictionary of sensor parameters used in the simulation.
            nr_workers (int): Number of workers involved.
            nr_visitors (int): Number of visitors involved.
            performance_metrics (dict): Dictionary of calculated performance metrics.
        """
        # Combine sensor parameters, worker/visitor counts, and performance metrics into a single dictionary
        performance_data = {
            **sensor_parameters,
            'nr_workers': nr_workers,
            'nr_visitors': nr_visitors,
            **performance_metrics
        }

        # Convert to DataFrame for saving
        performance_df = pd.DataFrame([performance_data])
        results_path = os.path.join('Results', 'metadata.csv')  # Define path for metadata CSV file

        if os.path.exists(results_path):
            # If file exists, append new data without headers
            performance_df.to_csv(results_path, mode='a', header=False, index=False)
        else:
            # If file doesn't exist, create it with headers
            performance_df.to_csv(results_path, mode='w', header=True, index=False)

    def load_ranges(self):
        """
        Load parameter ranges from the saved features configuration.

        Returns:
            dict: Dictionary containing parameter ranges for workers, visitors, and sensor settings.
        """
        # Retrieve ranges for sensors and environmental parameters from the saved features
        sensor_ranges = self.saved_features['sensitivity_analysis']['ranges']['sensor']
        worker_ranges = self.saved_features['sensitivity_analysis']['ranges']['environment']['nr_workers']
        visitor_ranges = self.saved_features['sensitivity_analysis']['ranges']['environment']['nr_visitors']

        # Construct and return a dictionary of the relevant parameter ranges
        ranges = {
            'nr_workers': worker_ranges,
            'nr_visitors': visitor_ranges,
            'P_u': sensor_ranges['P_u'],
            'P_f': sensor_ranges['P_f'],
            'std': sensor_ranges['std'],
        }
        return ranges

    def choose_parameters(self, parameters_to_search):
        """
        Choose which parameters to include in the grid search based on user input.

        Args:
            parameters_to_search (dict): Specifies whether to include each parameter in the grid search.

        Returns:
            dict: Dictionary containing chosen parameters and their corresponding ranges or default values.
        """
        # Load all available parameter ranges
        all_ranges = self.load_ranges()
        selected_ranges = {}  # Initialize dictionary to store selected parameter ranges

        # Loop through each parameter and determine if it should be included in the grid search
        for param, is_selected in parameters_to_search.items():
            if is_selected:
                # Include parameter range if selected
                selected_ranges[param] = all_ranges[param]
            else:
                # Otherwise, assign default values from the saved features configuration
                if param in self.saved_features['sensitivity_analysis']['default']['environment']:
                    default_value = self.saved_features['sensitivity_analysis']['default']['environment'][param]
                    selected_ranges[param] = [default_value]  # Wrap default value in a list
                elif param in self.saved_features['sensitivity_analysis']['default']['sensor']:
                    default_value = self.saved_features['sensitivity_analysis']['default']['sensor'][param]
                    selected_ranges[param] = [default_value]  # Wrap default value in a list

        return selected_ranges

    def run_grid_search(self, selected_parameters):
        """
        Perform grid search over the selected parameters with time estimation.

        Args:
            selected_parameters (dict): Dictionary of parameters and their ranges to search over.
        """
        # Extract parameter names and their corresponding values
        param_names = list(selected_parameters.keys())
        param_values = list(selected_parameters.values())

        # Generate all possible combinations of the parameter values using itertools.product
        param_combinations = list(product(*param_values))
        total_combinations = len(param_combinations)  # Get total number of parameter combinations

        # Print parameters that have a range (min, max, and number of values)
        print("\nParameters for grid search:")
        for param, values in selected_parameters.items():
            if len(values) > 1:  # Only print parameters that have a range (more than one value)
                print(f"Parameter: {param}, Min: {min(values)}, Max: {max(values)}, Number of values: {len(values)}")

        print(f"\nTotal combinations to search: {total_combinations}\n")


        # Initialize timing variables
        total_time = 0
        start_time = time.time()

        # Iterate through each combination and execute the simulation with the parameters
        for idx, combination in enumerate(param_combinations):
            param_dict = dict(zip(param_names, combination))  # Create dictionary of parameter-value pairs

            # Start time for this iteration
            iter_start_time = time.time()

            # Print the selected parameters in a concise way (optional)
            print(f"\nRunning Simulation {idx + 1}/{total_combinations} with parameters: {param_dict}")

            # Run the simulation with the chosen parameters (replace this with the actual simulation method)
            self.automated_main(**param_dict)

            # End time for this iteration
            iter_end_time = time.time()
            
            # Calculate iteration time and add it to the total time
            iter_time = iter_end_time - iter_start_time
            total_time += iter_time
            
            # Estimate remaining time
            avg_time_per_iteration = total_time / (idx + 1)  # Average time for completed iterations
            remaining_iterations = total_combinations - (idx + 1)
            estimated_remaining_time = remaining_iterations * avg_time_per_iteration

            # Print progress and time-related info
            print(f"Iteration {idx + 1}/{total_combinations} completed.")
            print(f"Time for this iteration: {iter_time:.2f} seconds.")
            print(f"Average time per iteration: {avg_time_per_iteration:.2f} seconds.")
            print(f"Estimated remaining time: {estimated_remaining_time / 60:.2f} minutes.\n")

        # End of grid search
        total_execution_time = time.time() - start_time
        print(f"\nGrid search completed in {total_execution_time / 60:.2f} minutes.")

    def automated_main(self, iterations=10, **parameters):
        """
        Automated version of the main simulation function.
        Runs the simulation with given inputs and stores results.

        Args:
            iterations (int): Number of iterations to run the simulation.
            **parameters: Other parameters such as sensor settings, worker and visitor numbers.
        """
        # Extract sensor parameters
        sensor_parameters = {
            "P_u": parameters.get("P_u"),
            "P_f": parameters.get("P_f"),
            "std": parameters.get("std"),
        }
        nr_workers = parameters.get("nr_workers")  # Number of workers
        nr_visitors = parameters.get("nr_visitors")  # Number of visitors

        # Initialize the simulator with the provided parameters
        simulator = Sim(sensor_parameters=sensor_parameters, nr_workers=nr_workers, nr_visitors=nr_visitors, runs=iterations)

        # Run the simulation
        print('\nRunning Simulations..\n')
        simulator.run()
        print('\nSimulations Complete..\n')

        # Calculate transition probabilities (P-matrix)
        print('\nCalculating P-matrix Average..\n')
        simulator.calc_transition_probabilities()
        print('\nP-matrix Average Calculated..\n')

        # Store simulation metadata
        self.metadata = pd.DataFrame(simulator.metadata)
        print('Metadata collected.\n')

        # Calculate performance metrics
        performance_metrics = self.calc_performance_metrics(self.metadata)

        # Store performance metrics in a CSV file
        self.store_performance_metrics(sensor_parameters, nr_workers, nr_visitors, performance_metrics)

    def save_performance_data(self):
        """
        Save the collected performance metrics to the 'metadata.csv' file.
        Appends new data to the existing file.
        """
        if self.performance_data.empty:
            print("No performance data to save.")
            return

        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

        # Save the performance data to the CSV file, either appending or creating the file
        if os.path.exists(self.filepath):
            self.performance_data.to_csv(self.filepath, mode='a', header=False, index=False)  # Append without header
        else:
            self.performance_data.to_csv(self.filepath, index=False)  # Write with header if file doesn't exist

        print(f"Performance data saved to {self.filepath}.")

    def get_performance_data(self):
        """
        Retrieve the saved performance data for further analysis.

        Returns:
            pd.DataFrame: DataFrame containing the performance data.
        """
        return self.performance_data



class Model:
    def __init__(self):
        self.df = pd.read_csv('Results/metadata.csv')
        """
        self.param_grid = {'alpha': np.linspace(0.001, 1, 10), 
                           'gamma': np.linspace(0.001, 1, 10)}  # You can adjust the range as needed 
        """
        self.param_grid = {
                    'alpha': [1e-3, 1e-2, 1e-1, 1],
                    'kernel': ['poly'],
                    'degree': [1, 2, 3, 4, 5]}


        self.modelname = None

        self.column_names = ['P_u', 'P_f', 'std', 'nr_workers', 'nr_visitors']

        self.input_dict = {'P_u': 1, 
                           'P_f': 1, 
                           'std': 1, 
                           'nr_workers': 1, 
                           'nr_visitors': 1}  # Removed 'Total Iterations'
        
        self.output_dict = {'Risk of Injury': 1}
        
        self.m = 1  # N.o. Outputs
        self.n = 5  # N.o. Inputs

    def select_input_features(self):
        """Ask the user to select which input features to use."""
        print("Please specify (0 or 1) whether you want to include each feature:")

        for feature in self.input_dict.keys():
            while True:
                try:
                    value = int(input(f"Include '{feature}': "))
                    if value in [0, 1]:
                        self.input_dict[feature] = value
                        break
                    else:
                        print("Please enter 0 or 1.")
                except ValueError:
                    print("Invalid input. Please enter 0 or 1.")

    def separate_features(self):
        """Separate input features (X) and output variables (y) based on selected inputs."""
        input_columns = [feature for feature, is_selected in self.input_dict.items() if is_selected == 1]
        self.X = self.df[input_columns]

        output_columns = [feature for feature, is_selected in self.output_dict.items() if is_selected == 1]
        self.y = self.df[output_columns]

        # Extract the Total Iterations column as weights
        self.weights = self.df['Total Iterations']  # Using it for weighting

    def fit_model(self):
        """Fit the Kernel Ridge model using the selected features and weights."""
        # Set up kernel regression with a Gaussian (RBF) kernel
        model = KernelRidge('poly')

        # Tune hyperparameters (e.g., bandwidth) using GridSearchCV
        grid_search = GridSearchCV(model, self.param_grid, cv=5)

        # Fit the model with weights
        grid_search.fit(self.X, self.y, sample_weight=self.weights)

        # Save the best model
        self.model = grid_search.best_estimator_  # Get the best model

        # Print the best degree and alpha
        best_params = grid_search.best_params_
        print(f"Best degree: {best_params['degree']}")
        print(f"Best alpha: {best_params['alpha']}")
    

    def generate_model_name(self):
        """Generate a model name based on the selected input features, date, and dataset size."""
        # Encoding the input features
        feature_encoding = [1 if is_selected else 0 for is_selected in self.input_dict.values()]
        
        # Convert indices to binary string
        param_encoding = ''.join(map(str, feature_encoding))
        
        # Get the current date and format it
        current_time = datetime.now().strftime('%y%m%d%H%M')
        
        # Get the number of rows in the dataset
        dataset_size = self.df.shape[0]
        
        # Create the model name
        self.modelname = f"kernel_{param_encoding}_{current_time}_{dataset_size}"

    def save_model(self):
        """Save the trained model to a file."""
        if self.model is not None:
            # Create the Models directory if it doesn't exist
            os.makedirs('Models', exist_ok=True)

            self.generate_model_name()  # Ensure the modelname is set before saving
            
            # Save the model to the Models directory
            model_path = os.path.join('Models', f'{self.modelname}.pkl')
            joblib.dump(self.model, model_path)
            print(f"Model saved as {model_path}")
        else:
            print("No model found to save...")

    def load_model(self):
        """Load a trained model from a specified filename."""
        try:
            # List all model files in the 'Models' directory
            model_files = [f for f in os.listdir('Models') if f.endswith('.pkl')]
            
            # If no models are found, notify the user
            if not model_files:
                print("No models found in the 'Models' directory.")
                return
            
            # Display the list of models
            print("Available models:")
            for i, model_file in enumerate(model_files):
                print(f"{i + 1}: {model_file}")
            
            # Ask user to select a model by number
            while True:
                try:
                    choice = int(input("Select a model number to load: ")) - 1
                    if 0 <= choice < len(model_files):
                        break
                    else:
                        print("Invalid choice. Please select a valid model number.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            # Load the selected model
            selected_model = model_files[choice]
            model_path = os.path.join('Models', selected_model)
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")


    def overview_data(self, num_bins=50):
        """
        Takes data from a DataFrame and shows normalized bin distribution for specified columns.
        
        Parameters:
        - num_bins: int, the number of bins to use for the histogram
        
        Returns:
        - A plot of the normalized bin distribution and a printout of the bin edges with sample counts.
        """
        
        fig, axs = plt.subplots(1, len(self.column_names), figsize=(15, 5), sharey=True)  # Common y-axis using sharey=True
        
        for ax_nr, column_name in enumerate(self.column_names):
            # Get the column data
            column_data = self.df[column_name]
            
            # Normalize the histogram by dividing by the total count
            axs[ax_nr].hist(column_data, bins=num_bins, edgecolor='black')  
            
            # Set plot details
            axs[ax_nr].set_title(f'{column_name}')
            axs[ax_nr].set_xlabel(column_name)

        # Set the common y-label
        fig.text(0.04, 0.5, 'Normalized Frequency', va='center', rotation='vertical')

        plt.tight_layout()
        plt.show()


    def visualize(self):
        if self.model is None:
            print("No model is available for visualization. Please fit or load a model first.")
            return

        self.X = self.df

        # Ask user for 2D or 3D plotting
        print('1. 2D Plot')
        print('2. 3D Plot')

        plot_type = input("Input choice: ").lower()

        # Collect available features
        available_features = [feature for feature, is_selected in self.input_dict.items() if is_selected == 1]

        if plot_type == '1':  # 2D plot
            if len(available_features) < 2:
                print("At least 2 selected features required for 2D plotting...")
                return

            print(f"Available features for 2D plot: {available_features}")
            feature_dict = {'x': None}

            # Let the user choose a feature for the x-axis
            while True:
                try:
                    print('Choose feature along x-axis:')
                    for nr, feature in enumerate(available_features):
                        print(f'{nr + 1}. {feature}')

                    feature_dict['x'] = available_features[int(input("x-axis: ")) - 1]
                    break
                except (ValueError, IndexError):
                    print("Invalid input...")

            if feature_dict['x'] not in available_features:
                print("Invalid feature selection.")
                return

            # Select features that will remain constant
            constant_features = [feature for feature in available_features if feature not in feature_dict.values()]

            # Ask the user to provide constant values for unselected features
            constants = {}
            for feature in constant_features:
                # Get the column data for plotting histograms
                column_data = self.df[feature]

                # Ask for constant value
                constants[feature] = float(input(f"Provide a constant value for {feature}: "))

            # Generate a grid of values for the selected feature on the x-axis
            xx = np.linspace(self.X[feature_dict['x']].min(), self.X[feature_dict['x']].max(), 100)

            # Create an input dataframe for prediction with constant values for other features
            input_df = pd.DataFrame({feature_dict['x']: xx.ravel()})

            # Add constant features to the input_df
            for feature, value in constants.items():
                input_df[feature] = value

            # Make predictions using the model
            predictions = self.predict(input_df)
            if predictions is None:
                return

            # Extract prediction results
            risk_predictions = predictions['Risk of Injury'].values.reshape(xx.shape)

            # Create a gridspec layout for 2D plots
            fig = plt.figure(figsize=(10, 8))
             # Load visual settings with defaults
            self.visual_settings = saved_features['plotstyle_settings']
            plt.rcParams['font.family'] = self.visual_settings.get('font_family', 'serif')
            plt.rcParams['font.serif'] = [self.visual_settings.get('font_name', 'Times New Roman')]
            

            # Define axes for the first example
            ax1 = fig.add_axes([0.1, 0.6, 0.8, 0.35])  # [left, bottom, width, height]
            ax1.plot(xx, risk_predictions, label='Risk of Injury', color='blue')
            ax1.set_title('Risk of Injury')
            ax1.set_xlabel(feature_dict['x'])
            ax1.set_ylabel('Risk')
            ax1.grid(True)

            histogram_positions = [0.04, 0.28, 0.52, 0.76]  # Evenly spaced
            # Create separate axes for histograms, one for each constant feature

            for i, feature in enumerate(constant_features):
                ax = fig.add_axes([histogram_positions[i], 0.1, 0.2, 0.35])  # Adjust position for the lower row
                ax.set_title(f'H{i + 1}')  # Histogram placeholder
                column_data = self.df[feature]
                ax.hist(column_data, bins=50, edgecolor='black', alpha=0.5, label=feature)
                ax.axvline(x=constants[feature], color='red', linewidth=2, label='Constant Value')
                ax.set_title(f'{feature}={constants[feature]}')
                ax.grid(True)

            plt.tight_layout()
            plt.show()

        elif plot_type == '2':  # 3D plot
            if len(available_features) < 3:
                print("At least 3 selected features required for 3D plotting...")
                return

            print(f"Available features for 3D plot: {available_features}")
            feature_dict = {'x': None, 'y': None}

            # Get features for x and y axes
            for axis in ['x', 'y']:
                while True:
                    try:
                        print(f'Choose feature along {axis}-axis:')
                        for nr, feature in enumerate(available_features):
                            print(f'{nr + 1}. {feature}')

                        feature_dict[axis] = available_features[int(input(f"{axis}-axis: ")) - 1]
                        break
                    except (ValueError, IndexError):
                        print("Invalid input...")

            if feature_dict['x'] not in available_features or feature_dict['y'] not in available_features:
                print("Invalid feature selection.")
                return

            # Select features that will remain constant
            constant_features = [feature for feature in available_features if feature not in feature_dict.values()]

            # Ask the user to provide constant values for unselected features
            constants = {}
            for feature in constant_features:
                # Get the column data for plotting histograms
                column_data = self.df[feature]

                # Ask for constant value
                constants[feature] = float(input(f"Provide a constant value for {feature}: "))

            # Generate a grid of values for the selected features
            x_range = np.linspace(self.X[feature_dict['x']].min(), self.X[feature_dict['x']].max(), 20)
            y_range = np.linspace(self.X[feature_dict['y']].min(), self.X[feature_dict['y']].max(), 20)

            xx, yy = np.meshgrid(x_range, y_range)

            # Create input dataframe for prediction with constant values for other features
            input_df = pd.DataFrame({feature_dict['x']: xx.ravel(), feature_dict['y']: yy.ravel()})
            for feature, value in constants.items():
                input_df[feature] = value

            # Make predictions using the model
            predictions = self.predict(input_df)
            if predictions is None:
                return

            risk_predictions = predictions['Risk of Injury'].values.reshape(xx.shape)

            # Create a gridspec layout for 3D plots
            fig = plt.figure(figsize=(12, 10))
             # Load visual settings with defaults
            self.visual_settings = saved_features['plotstyle_settings']
            plt.rcParams['font.family'] = self.visual_settings.get('font_family', 'serif')
            plt.rcParams['font.serif'] = [self.visual_settings.get('font_name', 'Times New Roman')]
        
        
            # Create the first axis for Risk of Injury
            ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.5], projection='3d')
            ax1.plot_surface(xx, yy, risk_predictions, cmap='coolwarm', edgecolor='none')
            ax1.set_title('Risk of Injury')
            ax1.set_xlabel(feature_dict['x'])
            ax1.set_ylabel(feature_dict['y'])
            ax1.set_zlabel('Risk')


            histogram_positions = [0.1, 0.4, 0.7]   # Evenly spaced
            # Create separate axes for histograms, one for each constant feature

            for i, feature in enumerate(constant_features):
                ax = fig.add_axes([histogram_positions[i], 0.1, 0.25, 0.2])  # Adjust position for the lower row
                ax.set_title(f'H{i + 1}')  # Histogram placeholder
                column_data = self.df[feature]
                ax.hist(column_data, bins=50, edgecolor='black', alpha=0.5, label=feature)
                ax.axvline(x=constants[feature], color='red', linewidth=2, label='Constant Value')
                ax.set_title(f'{feature}={constants[feature]}')
                ax.grid(True)
     

            plt.tight_layout()
            plt.show()

        else:
            print("Invalid plot type selection. Please choose either '2d' or '3d'.")

    def predict(self, input_df):
        """Make predictions using the loaded model and provided input DataFrame."""
        if self.model is None:
            print("Model not loaded. Please load a model before making predictions.")
            return None
        
        # Select relevant columns from the input DataFrame based on the input_dict
        input_columns = [feature for feature, is_selected in self.input_dict.items() if is_selected == 1]
        
        # Ensure the input DataFrame has the necessary columns
        missing_cols = set(input_columns) - set(input_df.columns)
        if missing_cols:
            print(f"Input DataFrame is missing columns: {missing_cols}")
            return None
        
        # Select only the necessary input features
        X_input = input_df[input_columns]
        
        # Make predictions
        predictions = self.model.predict(X_input)
        
        # Create a DataFrame for the predictions
        prediction_df = pd.DataFrame(predictions, columns=self.output_dict.keys())
        
        return prediction_df



    def compute_jacobian(self, input_df, epsilon=1e-4):
        """
        Computes the Jacobian matrix of the model's predictions with respect to each input parameter.
        
        Parameters:
        - X: np.ndarray, input data matrix (each row is an input configuration)
        - epsilon: float, the small perturbation to apply for numerical differentiation
        
        Returns:
        - jacobian: np.ndarray, Jacobian matrix (n_samples, n_features)
        """
        if self.model is None:
            raise ValueError("No model found. Train a model first.")
        
        # Convert DataFrame to NumPy array
        current_point = input_df.to_numpy()
    
        
        
        # Initialize the Jacobian matrix
        jacobian = np.zeros((self.m, self.n))
        
        # Base predictions without perturbation
        base_predictions = self.model.predict(current_point)
        
        # For each feature, perturb the values slightly and calculate the change in predictions
        for j in range(self.n):
            # Perturb the j-th feature
            perturbed_point = current_point.copy()
            perturbed_point[:, j] += epsilon
            
            # Predict with perturbed inputs
            perturbed_predictions = self.model.predict(perturbed_point)
            
            # Calculate the partial derivative (sensitivity) for the j-th feature
            jacobian[:, j] = (perturbed_predictions - base_predictions) / epsilon
        
        return jacobian
    
    def plot_jacobian_heatmap(self, jacobian):
        """Plots a heatmap for the Jacobian matrix with annotations."""
        
        plt.figure(figsize=(10, 6))
        # Load visual settings with defaults
        self.visual_settings = saved_features['plotstyle_settings']
        plt.rcParams['font.family'] = self.visual_settings.get('font_family', 'serif')
        plt.rcParams['font.serif'] = [self.visual_settings.get('font_name', 'Times New Roman')]
        
        sns.heatmap(jacobian, annot=True, cmap='coolwarm', fmt='.2f', 
                    xticklabels=self.input_dict.keys(), 
                    yticklabels=self.output_dict.keys())
        plt.title('Jacobian Heatmap: Sensitivity Analysis')
        plt.show()

    def evaluate_model(self):
        """
        Evaluates the model by calculating the Mean Squared Error (MSE) between the predicted 
        and true output values for the entire dataset.
        
        Returns:
        - mse_dict: dict, Mean Squared Error for each output variable (Risk of Injury and Average Productivity).
        """
        if self.model is None:
            raise ValueError("No model found. Please train or load a model before evaluating.")
        

        self.X = self.df[self.input_dict.keys()]

        self.y = self.df[self.output_dict.keys()]


        # Make predictions for the entire dataset
        predictions = self.predict(self.X)
        
        if predictions is None:
            print("Model predictions failed. Ensure the dataset is correctly formatted.")
            return None

        # Calculate MSE for each output variable
        mse_dict = {}
        for output in self.output_dict.keys():
            true_values = self.y[output].values
            predicted_values = predictions[output].values
            mse = mean_squared_error(true_values, predicted_values)
            mse_dict[output] = mse

        # Return MSE results
        return mse_dict
