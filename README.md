Robot Simulation and Analysis Tool
==================================

This tool provides an interactive platform to simulate robot behavior, analyze human movement patterns, assess sensor accuracy, and conduct comprehensive risk analysis using a Monte Carlo approach with Markov Chain estimation.

Installation
------------
1. Ensure Python is installed on your machine.
2. Clone the repository and navigate to the project folder:
   git clone https://github.com/your-repo/robot-simulation
   cd robot-simulation

3. Install the required dependencies using the following command:
   pip install -r requirements.txt

File Structure
--------------
The folder structure of the project is organized as follows:

.
├── Assets
├── Models
├── Results
├── analysis.py
├── main.py
├── README.txt
├── settings.py
└── simulation.py

- **Assets/**: Stores visual and auxiliary assets used in the simulation.
- **Models/**: Contains pre-trained models and output data.
- **Results/**: Stores simulation results, sensitivity analysis outputs, and generated videos.
- **analysis.py**: Contains code for video analysis, sensitivity studies, and predictions.
- **main.py**: The main entry point for running the program.
- **settings.py**: Stores saved features and pre-configured simulation settings.
- **simulation.py**: Manages the simulation environment and interactions.

Main Features
-------------
1. **Automated Sensitivity Analysis**: Automatically runs a grid search over selected parameters.
2. **Manual Simulation**: Users can manually input the number of workers, visitors, and other parameters for running the simulation.
3. **Sensitivity Analysis**: Load models, visualize them, and compute gradients for analysis.
4. **Prediction**: Load a pre-trained model, input data, and make predictions based on user-provided parameters.

Running the Program
-------------------
1. To run the tool, navigate to the main directory and execute the following:
   python main.py

2. Follow the on-screen instructions to choose between different modes of analysis:
   - **Automated Analysis**: Run a parameter sweep and analyze performance.
   - **Manual Simulation**: Configure and simulate the robot's interaction with people.
   - **Sensitivity Analysis**: Perform detailed analysis on key parameters and model gradients.
   - **Make Predictions**: Load a pre-trained model and make predictions based on the user's input.

Using the Modules
-----------------
- **analysis.py**: Use this module for video analysis, sensitivity analysis, and generating predictions. It provides functions to load models, run simulations, and evaluate performance.
- **simulation.py**: This module handles the core simulation logic, including human and robot movements, sensor accuracy, and results calculation.
- **settings.py**: Pre-configured settings for different sensors and simulation parameters are stored here. These settings can be modified to tailor the analysis to different environments.
- **main.py**: The main entry point, providing an interactive menu for running simulations and performing analyses.

Modularity
----------
The software allows for modularity in the environment, with many parameters easy to adjust. This flexibility enables tailoring the risk analysis to different facilities by changing the number of people, sensors, robot settings, and other environment variables.

Further Reading
---------------
For a detailed understanding of the methods used in this tool, including the Monte Carlo simulation approach and the Markov Chain estimation, refer to the accompanying report, available in the project repository.
