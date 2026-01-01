"""
================================================================================
PARTIE 1 - Génération de texte avec LSTM (Strong Training)
================================================================================
USAGE:
    Entraînement:
        python lstm_model.py --trainEval train --max_epochs 15000 --clip 5.0 --dropout 0.3
    Évaluation:
        python lstm_model.py --trainEval eval --length 300
================================================================================
"""

import unidecode
import string
import time
import math
from os import path, makedirs
from os.path import join
import torch
import torch.nn as nn
from argparse import ArgumentParser

# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)

# ------------------------------------------------------------------
# Characters
# ------------------------------------------------------------------
all_characters = string.printable
n_characters = len(all_characters)

# ------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------
lr = 0.005

# ------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------
def char_tensor(s):
    t = torch.zeros(len(s)).long()
    for i, ch in enumerate(s):
        t[i] = all_characters.index(ch)
    return t.to(device)

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {int(s)}s"

# ------------------------------------------------------------------
# Sequential training set (FULL TEXT, ordered)
# ------------------------------------------------------------------
def sequential_training_set(text, max_len=3000):
    inp = char_tensor(text[:-1])
    target = char_tensor(text[1:])
    return inp[:max_len], target[:max_len]

# ------------------------------------------------------------------
# Evaluation (text generation)
# ------------------------------------------------------------------
def evaluate(model, prime_str, predict_len=300, temperature=0.8):
    model.eval()
    hidden = model.init_hidden()

    prime_input = char_tensor(prime_str)
    predicted = prime_str

    for i in range(len(prime_str) - 1):
        _, hidden = model(prime_input[i], hidden)

    inp = prime_input[-1]

    for _ in range(predict_len):
        output, hidden = model(inp, hidden)
        output_dist = output.squeeze().div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        char = all_characters[top_i]
        predicted += char
        inp = char_tensor(char)

    return predicted

# ------------------------------------------------------------------
# LSTM Model
# ------------------------------------------------------------------
class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            n_layers,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x = x.view(1, -1)
        x = self.encoder(x)
        output, hidden = self.lstm(x, hidden)
        output = self.dropout(output)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return (
            torch.zeros(self.n_layers, 1, self.hidden_size, device=device),
            torch.zeros(self.n_layers, 1, self.hidden_size, device=device),
        )

# ------------------------------------------------------------------
# Training step
# ------------------------------------------------------------------
def train_step(model, inp, target, optimizer, criterion, clip=5.0):
    model.train()
    hidden = model.init_hidden()
    optimizer.zero_grad()

    loss = 0
    for i in range(inp.size(0)):
        output, hidden = model(inp[i], hidden)
        loss += criterion(output, target[i].unsqueeze(0))

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    return loss.item() / inp.size(0)

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--trainingData", default="data/shakespeare.txt")
    parser.add_argument("--trainEval", default="train", choices=["train", "eval"])
    # Accept both --model and --model_dir for backward compatibility
    parser.add_argument("-m", "--model", "--model_dir", dest="model_dir", default="models/lstm",
                        help="Directory where model files are stored (default: models/lstm)")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max_epochs", type=int, default=15000)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--length", type=int, default=300)

    args = parser.parse_args()

    # Load text
    text = unidecode.unidecode(open(args.trainingData, encoding="utf-8").read())
    print("Corpus size:", len(text))

    # Model
    model = LSTMModel(
        n_characters,
        args.hidden_size,
        n_characters,
        args.num_layers,
        args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model_name = f"lstm_{args.num_layers}_{args.hidden_size}.pt"
    if not path.exists(args.model_dir):
        makedirs(args.model_dir)

    # -------------------- TRAIN --------------------
    if args.trainEval == "train":
        start = time.time()
        inp, target = sequential_training_set(text)

        for epoch in range(1, args.max_epochs + 1):
            loss = train_step(model, inp, target, optimizer, criterion, args.clip)
            if epoch % 500 == 0:
                print(f"[{epoch}/{args.max_epochs}] loss={loss:.4f} time={time_since(start)}")

        torch.save(model, join(args.model_dir, model_name))
        print("Model saved:", join(args.model_dir, model_name))

    # -------------------- EVAL --------------------
    else:
        # Try to locate the model file with a few fallbacks to handle different naming conventions
        def _find_model_file(model_dir, expected_name):
            from glob import glob
            expected_path = join(model_dir, expected_name)
            if path.exists(expected_path):
                return expected_path
            # look for files like 'lstm*_{num_layers}_{hidden}.pt' inside model_dir
            pattern = join(model_dir, f"lstm*_{args.num_layers}_{args.hidden_size}.pt")
            matches = glob(pattern)
            if matches:
                return matches[0]
            # look in parent 'models' folder as a last resort
            parent = path.dirname(model_dir)
            pattern2 = join(parent, f"lstm*_{args.num_layers}_{args.hidden_size}.pt")
            matches = glob(pattern2)
            if matches:
                return matches[0]
            return None

        model_path = _find_model_file(args.model_dir, model_name)
        if model_path is None:
            raise FileNotFoundError(
                f"Model file not found: expected {join(args.model_dir, model_name)}.\n"
                f"Checked {args.model_dir} and its parent directory for matching files."
            )
        if model_path != join(args.model_dir, model_name):
            print("Warning: expected model not found; using alternative file:", model_path)

        model = torch.load(
            model_path,
            map_location=device,
            weights_only=False
        )

        start_prompt = input(
            "\nEnter starting text (letter / word / sentence): "
        ).strip()

        if len(start_prompt) == 0:
            start_prompt = "Th"

        print("\nGenerated text:\n")
        print(evaluate(model, start_prompt, args.length))
