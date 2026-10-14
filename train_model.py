
import csv
import os
import random
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier

DATA_FILE = "game_data.csv"
MODEL_FILE = "rps_ann.pkl"

MOVES = ["Rock", "Paper", "Scissors"]


def encode_history(history):
    values = []

    for move in history:
        for possible_move in MOVES:
            values.append(1 if move == possible_move else 0)

    return values


def generate_dataset():
    random.seed(42)

    with open(DATA_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["prev1", "prev2", "prev3", "next_move"])

        strategies = ["repeat", "cycle", "counter", "random"]

        for _ in range(300):
            strategy = random.choice(strategies)
            history = []

            for _ in range(20):
                if not history:
                    next_move = random.choice(MOVES)

                elif strategy == "repeat":
                    next_move = history[-1]

                elif strategy == "cycle":
                    next_move = MOVES[
                        (MOVES.index(history[-1]) + 1) % 3
                    ]

                elif strategy == "counter":
                    last_move = history[-1]
                    next_move = MOVES[
                        (MOVES.index(last_move) + 1) % 3
                    ]

                else:
                    next_move = random.choice(MOVES)

                previous = [None] * (3 - len(history[-3:])) + history[-3:]
                writer.writerow([
                    previous[0] or "",
                    previous[1] or "",
                    previous[2] or "",
                    next_move
                ])

                history.append(next_move)


def train_model():
    features = []
    labels = []

    with open(DATA_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            history = [
                row["prev1"] or None,
                row["prev2"] or None,
                row["prev3"] or None
            ]

            features.append(encode_history(history))
            labels.append(row["next_move"])

    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42
    )

    model.fit(np.array(features), labels)

    with open(MODEL_FILE, "wb") as file:
        pickle.dump(model, file)

    print("ANN training completed!")
    print("Training examples:", len(features))
    print("Model saved as:", MODEL_FILE)



if __name__ == "__main__":
    if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
        generate_dataset()
        print("Synthetic dataset created.")

    train_model()
