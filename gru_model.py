"""
================================================================================
PARTIE 1 - Génération de texte avec GRU (Strong & Clean)
================================================================================
USAGE:
    Entraînement:
        python gru_model.py --trainEval train --max_epochs 20000 --clip 5.0 --dropout 0.3
    Évaluation:
        python gru_model.py --trainEval eval --length 300
================================================================================
"""

import unidecode
import string
import random
import time, math
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
# Characters (character-level model)
# ------------------------------------------------------------------
all_characters = string.printable
n_characters = len(all_characters)
# ------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------
lr = 0.003
chunk_len = 80   # أطول من RNN العادي = تعلم سياق أفضل

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
# Random training chunks (stochastic training)
# ------------------------------------------------------------------
def random_chunk(text):
    start = random.randint(0, len(text) - chunk_len - 1)
    end = start + chunk_len + 1
    return text[start:end]

def random_training_set(text):
    chunk = random_chunk(text)
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target

# ------------------------------------------------------------------
# GRU Model
# ------------------------------------------------------------------
class GRUModel(nn.Module):
    """
    Architecture:
        input char index
            -> Embedding (Encoder)
            -> GRU
            -> Dropout
            -> Linear (Decoder)
    """

    def __init__(self, input_size, hidden_size, output_size, n_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # Encoder: index -> vector
        self.encoder = nn.Embedding(input_size, hidden_size)

        # GRU
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            dropout=dropout if n_layers > 1 else 0.0
        )

        # Regularization
        self.dropout = nn.Dropout(dropout)

        # Decoder: hidden -> character probabilities
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        """
        x: character index (scalar)
        hidden: (n_layers, 1, hidden_size)
        """

        # x.shape: []
        x = x.view(1, -1)              # (1,1)

        x = self.encoder(x)            # (1,1,hidden_size)

        output, hidden = self.gru(x, hidden)
        # output.shape = (1,1,hidden_size)
        # hidden.shape = (n_layers,1,hidden_size)

        output = self.dropout(output)

        output = self.decoder(output.view(1, -1))
        # output.shape = (1,n_characters)

        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size, device=device)

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
# Text generation
# ------------------------------------------------------------------
def evaluate(model, start="Th", length=300, temperature=0.8):
    model.eval()
    hidden = model.init_hidden()

    start_input = char_tensor(start)
    predicted = start

    for i in range(len(start) - 1):
        _, hidden = model(start_input[i], hidden)

    inp = start_input[-1]

    for _ in range(length):
        output, hidden = model(inp, hidden)
        output_dist = output.squeeze().div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        char = all_characters[top_i]
        predicted += char
        inp = char_tensor(char)

    return predicted

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--trainingData", default="data/shakespeare.txt")
    parser.add_argument("--trainEval", default="train", choices=["train", "eval"])
    parser.add_argument("--model_dir", default="models/gru")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max_epochs", type=int, default=8000)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--length", type=int, default=300)

    args = parser.parse_args()

    text = unidecode.unidecode(open(args.trainingData, encoding="utf-8").read())
    print("Corpus size:", len(text))

    model = GRUModel(
        n_characters,
        args.hidden_size,
        n_characters,
        args.num_layers,
        args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if not path.exists(args.model_dir):
        makedirs(args.model_dir)

    model_name = f"gru_{args.num_layers}_{args.hidden_size}.pt"

    # ---------------- TRAIN ----------------
    if args.trainEval == "train":
        start = time.time()
        for epoch in range(1, args.max_epochs + 1):
            inp, target = random_training_set(text)
            loss = train_step(model, inp, target, optimizer, criterion, args.clip)

            if epoch % 500 == 0:
                print(f"[{epoch}/{args.max_epochs}] loss={loss:.4f} time={time_since(start)}")

        torch.save(model, join(args.model_dir, model_name))
        print("Model saved:", join(args.model_dir, model_name))

    # ---------------- EVAL ----------------
    else:
        model = torch.load(
        join(args.model_dir, model_name),
        map_location=device,
        weights_only=False
    )
    model.eval()

    print("\n================ TEXT GENERATION (GRU) ================\n")

    # إدخال من المستخدم
    start_text = input(
        "Enter starting text (letter / word / sentence): "
    ).strip()

    # إذا المستخدم ضغط Enter بدون كتابة شيء
    if len(start_text) == 0:
        start_text = "Th"

    print("\nGenerated text:\n")
    print(evaluate(model, start_text, args.length))

    print("\n======================================================\n")
