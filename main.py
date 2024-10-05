# -----------------------------------------------------------------------------
# Author: Ludwig Horvath
# Email: ludhor@kth.se
# Date: 2024-10-05
# -----------------------------------------------------------------------------


# Local modules
from analysis import Video, Sensitivity_analysis, Model
from simulation import Sim
from settings import saved_features
import pandas as pd



def main():

    # Program introduction
    print("\nWelcome to the Robot Simulation and Analysis Tool.")
    print("This tool allows you to simulate movements, run sensitivity analysis, and generate visualizations.\n")
    
    while True:
        # High-level menu for choosing analysis mode
        print("Please choose an action:")
        print("1. Automated Analysis")
        print("2. Manual Simulation")
        print("3. Sensitivity Analysis")
        print("4. Make Predictions")
        print("5. Exit")
        
        mode_choice = input("Enter your choice (1, 2, 3, 4, or 5): ")
        print()  # Blank line for better readability
        
        if mode_choice == '1':
            automated_analysis()
        elif mode_choice == '2':
            manual_analysis()
        elif mode_choice == '3':
            sensitivity_analysis()
        elif mode_choice == '4':
            make_predictions()
        elif mode_choice == '5':
            print("Exiting the program...\n")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, or 5.\n")



def automated_analysis():
    """Handles the automated sensitivity analysis mode"""
    print("Starting Automated Analysis...\n")
    analysis = Sensitivity_analysis()

    # Define which parameters to include in the grid search
    parameters_to_search = {
        'nr_workers': True,   # Include workers in the grid search
        'nr_visitors': True,  # Include visitors
        'P_u': True,              
        'P_f': True,        
        'std': True      # Include accuracy
    }

    # Load selected ranges and run the analysis
    selected_parameters = analysis.choose_parameters(parameters_to_search)
    analysis.run_grid_search(selected_parameters)

    print("Automated Analysis Complete. Returning to main menu...\n")

def manual_analysis():
    """Handles the manual analysis mode where users input parameters"""

    sensor_indices = [None]

    for nr, sensor_type in enumerate(saved_features['sensors'].keys()):
        print(f'{nr+1}. {sensor_type}: {saved_features["sensors"][sensor_type]}')
        sensor_indices.append(sensor_type)

    nr_choice = int(input('Enter number corresponding to sensor: '))
    sensor_parameters = saved_features['sensors'][sensor_indices[nr_choice]]
    

    # Get user inputs for the simulation parameters
    print("Please provide the following parameters to configure the simulation:\n")
    speed = int(get_user_input("Enter animation frame interval (between 10 and 400 ms): ", 10, 400))
    iterations = int(get_user_input("Enter number of iterations (between 1 and 100): ", 1, 100))
    nr_workers = int(get_user_input("Enter number of workers (between 0 and 5): ", 0, 5))
    nr_visitors = int(get_user_input("Enter number of visitors (between 0 and 5): ", 0, 5))

    # Initialize the simulation
    simulator = Sim(sensor_parameters=sensor_parameters, nr_workers=nr_workers, nr_visitors=nr_visitors, runs=iterations)

    # Run the simulation and calculations
    print('\nRunning Simulations..\n')
    simulator.run()
    print('\nSimulations Complete..\n')

    # Video object placeholder (None initially)
    video = None

    # Menu for user interaction after simulation is complete
    while True:
        print("Choose an option:")
        print("1. Save Model")
        print("2. Play video")
        print("3. Save Results (Video, Model, or Both)")
        print("4. Clear video")
        print("5. Return to Main Menu")
        
        choice = input("Enter your choice (1, 2, 3, 4, or 5): ")
        print()  # Blank line for better readability
        
        if choice == '1':
            simulator.save_calc_model()
            print("Model saved successfully.\n")
        
        elif choice == '2':
            video = handle_video_playback(simulator, video, speed)
        
        elif choice == '3':
            handle_saving(simulator, video)
        
        elif choice == '4':
            video = clear_video(video)

        elif choice == '5':
            print("Returning to Main Menu...\n")
            break  # Return to the main menu

        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, or 5.\n")

def make_predictions():
    """Function to load a model, get input data from the user, and make predictions."""
    # Load the model
    my_model = Model()
    my_model.load_model()

    # Create an input dictionary based on required input features
    input_dict = {
        'P_u': get_user_input("Enter the value for P_u (0 to 1): ", 0, 1),
        'P_f': get_user_input("Enter the value for P_f (0 to 1): ", 0, 1),
        'std': get_user_input("Enter the value for standard deviation (0 to 1): ", 0, 1),
        'nr_workers': get_user_input("Enter the number of workers (0 to 5): ", 0, 5),
        'nr_visitors': get_user_input("Enter the number of visitors (0 to 5): ", 0, 5),
    }

    # Convert the input into a DataFrame for prediction
    input_df = pd.DataFrame([input_dict])

    # Make predictions
    predictions = my_model.predict(input_df)

    # Display the predictions
    if predictions is not None:
        print("Predictions:")
        print(predictions)
    else:
        print("No predictions made.")

def sensitivity_analysis():
    """Handles Sensitivity Analysis including model creation, loading, and evaluation"""
    print("Starting Sensitivity Analysis...\n")

    while True:
        print("Please choose an action:")
        print("1. Create and Save Model")
        print("2. Load Model")
        print("3. Visualize Model")
        print("4. Calculate Jacobian and Visualize Gradient")
        print("5. Evaluate Model")
        print("6. Return to Main Menu")

        action_choice = input("Enter your choice (1, 2, 3, 4, 5, or 6): ")
        print()  # Blank line for better readability

        if action_choice == '1':
            create_and_save_model()
        elif action_choice == '2':
            load_model()
        elif action_choice == '3':
            visualize_model()
        elif action_choice == '4':
            calculate_jacobian()
        elif action_choice == '5':
            evaluate_model()
        elif action_choice == '6':
            print("Returning to Main Menu...\n")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.\n")

def create_and_save_model():
    """Creates a model instance, selects input features, and saves the model"""
    model_instance = Model()

    # Select input features based on simulated user input
    model_instance.select_input_features()
    model_instance.separate_features()
    model_instance.fit_model()
    model_instance.save_model()
    print("Model created and saved successfully.\n")

def load_model():
    """Loads a model instance"""
    new_model = Model()
    new_model.load_model()
    print("Model loaded successfully.\n")

def visualize_model():
    """Visualizes the loaded model"""
    new_model = Model()
    new_model.load_model()
    new_model.visualize()
    print("Model visualization complete.\n")

def calculate_jacobian():
    """Calculates the Jacobian and visualizes the gradient at a point"""
    # Prompt user for current state input
    current_state = {}
    current_state['P_u'] = get_user_input("Enter the value for P_u: ", 0, 1)  # Assuming P_u is a probability
    current_state['P_f'] = get_user_input("Enter the value for P_f: ", 0, 1)  # Assuming P_f is a probability
    current_state['std'] = get_user_input("Enter the value for standard deviation (0 to 1): ", 0, 1)
    current_state['nr_workers'] = get_user_input("Enter the number of workers (0 to 5): ", 0, 5)
    current_state['nr_visitors'] = get_user_input("Enter the number of visitors (0 to 5): ", 0, 5)

    input_df = pd.DataFrame([current_state])

    my_model = Model()
    my_model.load_model()

    # Call the predict method
    predictions = my_model.predict(input_df)

    jacobian = my_model.compute_jacobian(input_df)

    # Display the predictions
    if predictions is not None:
        print("Predictions:")
        print(predictions)

    print("Jacobian:")
    print(jacobian)

    my_model.plot_jacobian_heatmap(jacobian)
    print("Jacobian calculation and visualization complete.\n")

def evaluate_model():
    """Evaluates the loaded model"""
    my_model = Model()
    my_model.load_model()
    mse = my_model.evaluate_model()
    print(f"Model evaluation complete. MSE: {mse}\n")

def get_user_input(prompt, min_val, max_val):
    """Utility function to get validated user input within a range"""
    while True:
        try:
            value = float(input(prompt))  # Changed to float to accept decimal values
            if value < min_val or value > max_val:
                raise ValueError(f"Value must be between {min_val} and {max_val}.")
            return value
        except ValueError as e:
            print(e)

def handle_video_playback(simulator, video, speed):
    """Handles video playback based on user choice"""
    if video is not None:
        print("A video is already active. Please save or clear it before creating a new one.\n")
        return video

    available_videos = []
    if simulator.first_G is not None:
        available_videos.append('Goal')
    if simulator.first_I is not None:
        available_videos.append('Injury')

    if not available_videos:
        print("No videos available to play.\n")
        return video

    print("Available Videos:\n")
    for video_option in available_videos:
        print(f"* {video_option}")

    video_choice = input('Input Choice of Video to Play: ')

    if video_choice not in available_videos:
        print("Chosen Video Does not Exist.\n")
        return video

    # Create the video object based on user choice
    if video_choice == 'Goal':
        video = Video(data=simulator.dataG)
    elif video_choice == 'Injury':
        video = Video(data=simulator.dataI)

    video.interval = speed  # Set the speed for the animation
    video.create_animation()
    video.play_animation()
    print(f"{video_choice} video played successfully.\n")

    return video

def handle_saving(simulator, video):
    """Handles saving of model and/or video"""
    print("What would you like to save?")
    print("1. Model")
    print("2. Video")
    print("3. Both")
    
    save_choice = input("Enter your choice (1, 2, or 3): ")
    print('\nSaving...\n')
    
    if save_choice == '1':
        simulator.save_calc_model()
        print("Model saved successfully.\n")
    
    elif save_choice == '2':
        if video is not None:
            video.save_animation(case="Video")
            print("Video saved successfully.\n")
        else:
            print("No active video to save.\n")
    
    elif save_choice == '3':
        simulator.save_calc_model()
        print("Model saved successfully.\n")
        
        if video is not None:
            video.save_animation(case="Video")
            print("Video saved successfully.\n")
        else:
            print("No active video to save.\n")
    else:
        print("Invalid choice. Please enter 1, 2, or 3.\n")

def clear_video(video):
    """Clears the current video"""
    if video is not None:
        print("Current video cleared successfully.\n")
        return None
    else:
        print("No active video to clear.\n")
        return video

if __name__ == '__main__':
    main()
