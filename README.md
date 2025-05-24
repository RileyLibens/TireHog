Project Title: TireHog - AI for Machine Parameter Optimization

Overall Project Goal: Using AI to determine operating parameters/Active control for the TireHog tire recycling machine.
RNNarchitecture Branch Focus: This RNNarchitecture branch is dedicated to the development, testing, and refinement of a Recurrent Neural Network (RNN) model for predicting optimal operating parameters.

This is a collaborative project. Key contributions within this branch include:
Development and implementation of the core RNN model (see RNNarchitecture1.py, RNNarchitecture2.py) - Elliott Yankelevich
Data preprocessing pipeline for RNN input (DataToMatrix.py) - Elliott Yankelevich
Model training, evaluation, and iterative improvements, including addressing dummy data recognition (see commit history by Elliott Yankelevich) - Elliott Yankelevich
Initial data collection - Riley Libens

[RNN Model Implementation](RNNarchitecture2.py)
Technologies Used: Python, PyTorch, Pandas, NumPy, MatPlotLib

Brief Explanation of Key Files:
DataFromTireHog.xlsx - Uncleaned Data Sample from TireHog Machine.
CleanDataSample.xlsx - Cleaned Data Sample from TireHog Machine.
TireHog Dummy Data - Simulated data based off of Clean Data Sample.
DataToMatrix.py - PoC for importing Dummy Data from excel scheets to matrix format required by the RNN.
RNNarchitecture1.py - PoC for RNNarchitecture hyperparameter setup.
RNNarchitecture2.py - Finalized architecture with correct hyperparameters, fixed the program so it recognizes the new dummy data.
training_loss.png - Visualized training/validation loss from RNNarchitecture2.py model training.
results - Finalized results produced from running RNNarchitecture2.py.
