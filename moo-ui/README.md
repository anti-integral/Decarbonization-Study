# Multi-Objective Optimization for Renewable Energy

This package implements a multi-objective optimization tool for renewable energy systems. It allows users to upload data for renewable energy production, plant demand, and constants, then run optimizations to find optimal configurations balancing energy cost and CO2 emissions. The tool provides an interactive web interface for data input, visualization of results through Pareto fronts, and detailed analysis of selected solutions.

## Main Python Files

1. `application.py`: Contains the Dash application that creates the web interface, handles file uploads, and manages user interactions.
2. `main_opt.py`: Implements the core optimization logic using the NSGA2 algorithm from the pymoo library.
3. `utility.py`: Defines the `RenewableEnergyProblem` class and utility functions for data processing and result aggregation.

## Data CSV Files

The application requires three CSV files for input:

1. RE Production Data: Contains columns [datetime, Ppv, Pw, Pcs] for solar, wind, and concentrated solar power production.
2. Plant Demand Data: Contains columns [day, PF] for daily power demand.
3. Constants: Contains columns [Constant, Value] with various system parameters and coefficients.

## How to Run

### Without Dockerfile

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
2. Run the application:
   ```
   python application.py
   ```
3. Open a web browser and navigate to `http://127.0.0.1:8080`

### With Dockerfile

1. Build the Docker image:
   ```
   docker build -t moo-ui .
   ```
2. Run the container:
   ```
   docker run -p 8080:8080 moo-ui
   ```
3. Open a web browser and navigate to `http://127.0.0.1:8080`

