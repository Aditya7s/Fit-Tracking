# Barbell Exercise Classifier

This project aims to create Python scripts to process, visualize, and model accelerometer and gyroscope data to create a machine learning model that can classify barbell exercises and count repetitions.

## Motivation

Barbell exercises are a common form of strength training that involve lifting a weighted barbell. However, performing these exercises incorrectly can lead to injuries or suboptimal results. Therefore, it is important to monitor the quality and quantity of the exercises.

This project uses sensor data from a MetaWear sensor on a wrist to analyze the movement patterns and provide feedback to the user. The project also explores different machine learning techniques to classify the exercises and count the repetitions.

## Data

The data for this project comes from [Dave Ebbelaar]([https://www.readme-templates.com/](https://www.youtube.com/@daveebbelaar)). The data was collected from 5 participants performing different types of exercises: benchpress, deadlift, overhead press, bent-over rows, and squat. There is also data from the user just resting. The exercises were done at varying intensity as well, labeled as "heavy" and "medium", as well as performed sitting and standing.

## Scripts

The project contains the following Python files:

- `make_dataset.py`: This file reads the `.csv` files and cleans and compiles them together, putting in relevant data. It exports into a `.pkl` file.
- `visualize.py`: This file creates various plots to explore the data and understand the patterns and distributions. It outputs a folder with the generated graphs.
- `remove_outliers.py`: This file removes outliers using the IQR, Chauvanet's method, and Local Outlier Factors for each exercise. It outputs a `.pkl` file.
- `build_features.py`: This file filters the data to make it better for machine learning algorithms. It runs the data through low pass filters, performs principal component analysis, temporal abstraction, and also clusters it using k-means clustering. It outputs the data with all these features into a `.pkl` file.
- `count_repetitions.py`: This file uses a function to identify each repetition performed for each exercise set. This is done through analysis of the extrema in the data. It outputs the data into a `.pkl` file.
- `train_model.py`: This file searches and creates the best model to predict the exercise performed. It finds the best features using a grid search on the hyperparameters and features selection, then trains multiple models: A feedforward neural network, random forest, K Nearest Neighbor, Decision Tree, and Naive Bayes. It then selects the best model based off accuracy scores and then evaluates results into a confusion matrix.

## To Be Done

Export the trained model using a pipeline for later utilization, without retraining the model over and over.
