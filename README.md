# Rock, Paper, Scissors AI Game Using ANN

## Project Overview

This project is a graphical Rock, Paper, Scissors game developed using Python. It uses an Artificial Neural Network (ANN) built with Scikit-learn's MLPClassifier to predict the player's next move and select a counter-move.

## Technologies Used

- Python
- Tkinter
- NumPy
- Pandas
- Scikit-learn
- CSV for dataset storage

## Features

- Graphical user interface with three move buttons
- AI opponent powered by an ANN
- Synthetic training data generation
- Collection of real player moves
- Periodic model retraining
- Score tracking and game reset

## How to Run

1. Install Python.
2. Download or clone this repository.
3. Install the required libraries:

   `python -m pip install -r requirements.txt`

4. Train the model:

   `python train_model.py`

5. Start the game:

   `python game.py`

## How It Works

The ANN learns patterns from previous player moves. It predicts the player's next move, and the game selects the move that beats the prediction. Gameplay data is stored in a CSV file for retraining.

## Learning Objective

The project demonstrates basic machine learning, neural networks, data collection, model training, and Python GUI development.
