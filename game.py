
import csv
import os
import pickle
import tkinter as tk
from tkinter import messagebox

from train_model import (
    DATA_FILE,
    MODEL_FILE,
    MOVES,
    encode_history,
    generate_dataset,
    train_model
)

if not os.path.exists(DATA_FILE):
    generate_dataset()

if not os.path.exists(MODEL_FILE):
    train_model()

with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)


history = []
player_score = 0
ai_score = 0
draws = 0
rounds = 0


def padded_history():
    return [None] * (3 - len(history)) + history[-3:]


def get_ai_move():
    features = encode_history(padded_history())
    predicted_move = model.predict([features])[0]

    counter_moves = {
        "Rock": "Paper",
        "Paper": "Scissors",
        "Scissors": "Rock"
    }

    return counter_moves[predicted_move], predicted_move


def save_player_move(previous, current):
    with open(DATA_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(previous + [current])


def determine_winner(player, ai):
    if player == ai:
        return "Draw"

    if (
        (player == "Rock" and ai == "Scissors")
        or (player == "Paper" and ai == "Rock")
        or (player == "Scissors" and ai == "Paper")
    ):
        return "You win!"

    return "AI wins!"


def play(player_move):
    global player_score, ai_score, draws, rounds, model

    previous = padded_history()

    ai_move, predicted_move = get_ai_move()

    result = determine_winner(player_move, ai_move)

    if result == "You win!":
        player_score += 1
    elif result == "AI wins!":
        ai_score += 1
    else:
        draws += 1

    rounds += 1

    save_player_move(previous, player_move)
    history.append(player_move)

    if len(history) > 3:
        history.pop(0)

    player_label.config(text=f"Your move: {player_move}")
    ai_label.config(text=f"AI move: {ai_move}")
    prediction_label.config(
        text=f"ANN predicted you might play: {predicted_move}"
    )
    result_label.config(text=result)

    score_label.config(
        text=f"You: {player_score}    AI: {ai_score}    Draws: {draws}"
    )

    data_label.config(text=f"Moves collected: {rounds}")

    if rounds % 10 == 0:
        result_label.config(text="Training ANN with new data...")

        root.update_idletasks()

        try:
            train_model()

            with open(MODEL_FILE, "rb") as file:
                model = pickle.load(file)

            result_label.config(
                text=f"{result} | ANN retrained with new data!"
            )
        except Exception as error:
            result_label.config(
                text=f"{result} | Retraining error: {error}"
            )


def reset_game():
    global player_score, ai_score, draws, rounds

    history.clear()
    player_score = 0
    ai_score = 0
    draws = 0
    rounds = 0

    player_label.config(text="Your move: -")
    ai_label.config(text="AI move: -")
    prediction_label.config(text="ANN prediction: -")
    result_label.config(text="Choose your move!")
    score_label.config(text="You: 0    AI: 0    Draws: 0")
    data_label.config(text="Moves collected: 0")


def retrain_now():
    global model

    try:
        train_model()

        with open(MODEL_FILE, "rb") as file:
            model = pickle.load(file)

        messagebox.showinfo(
            "ANN Training",
            "The ANN has been retrained using the current dataset."
        )
    except Exception as error:
        messagebox.showerror("Training Error", str(error))


root = tk.Tk()
root.title("Rock Paper Scissors - ANN AI")
root.geometry("520x540")
root.resizable(False, False)
root.configure(bg="#172033")

title_label = tk.Label(
    root,
    text="ROCK  PAPER  SCISSORS",
    font=("Arial", 19, "bold"),
    bg="#172033",
    fg="white"
)
title_label.pack(pady=20)

subtitle_label = tk.Label(
    root,
    text="Play against an Artificial Neural Network",
    font=("Arial", 10),
    bg="#172033",
    fg="#B8C7E0"
)
subtitle_label.pack()

button_frame = tk.Frame(root, bg="#172033")
button_frame.pack(pady=25)

for move, symbol in [
    ("Rock", "✊"),
    ("Paper", "✋"),
    ("Scissors", "✌")
]:
    button = tk.Button(
        button_frame,
        text=f"{symbol}\n{move}",
        font=("Arial", 12, "bold"),
        width=10,
        height=3,
        command=lambda m=move: play(m),
        bg="#3B82F6",
        fg="white",
        activebackground="#2563EB",
        activeforeground="white",
        relief="flat",
        cursor="hand2"
    )
    button.pack(side="left", padx=5)

player_label = tk.Label(
    root, text="Your move: -",
    font=("Arial", 12),
    bg="#172033", fg="white"
)
player_label.pack(pady=5)

ai_label = tk.Label(
    root, text="AI move: -",
    font=("Arial", 12),
    bg="#172033", fg="white"
)
ai_label.pack(pady=5)

prediction_label = tk.Label(
    root, text="ANN prediction: -",
    font=("Arial", 10),
    bg="#172033", fg="#93C5FD"
)
prediction_label.pack(pady=5)

result_label = tk.Label(
    root, text="Choose your move!",
    font=("Arial", 15, "bold"),
    bg="#172033", fg="#FBBF24"
)
result_label.pack(pady=12)

score_label = tk.Label(
    root, text="You: 0    AI: 0    Draws: 0",
    font=("Arial", 12, "bold"),
    bg="#172033", fg="white"
)
score_label.pack(pady=5)

data_label = tk.Label(
    root, text="Moves collected: 0",
    font=("Arial", 10),
    bg="#172033", fg="#B8C7E0"
)
data_label.pack(pady=5)

bottom_frame = tk.Frame(root, bg="#172033")
bottom_frame.pack(pady=15)

tk.Button(
    bottom_frame,
    text="Reset Game",
    command=reset_game,
    font=("Arial", 10),
    width=13
).pack(side="left", padx=5)

tk.Button(
    bottom_frame,
    text="Retrain ANN",
    command=retrain_now,
    font=("Arial", 10),
    width=13
).pack(side="left", padx=5)

root.mainloop()
