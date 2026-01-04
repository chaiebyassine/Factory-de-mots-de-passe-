"""
================================================================================
PARTIE 1 - Génération de texte avec Simple RNN (Strong Training)
================================================================================
USAGE:
    Entraînement:
        python rnn_model.py --trainEval train --max_epochs 8000 --clip 5.0
    Évaluation:
        python rnn_model.py --trainEval eval --length 300
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
chunk_len = 80
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
# Random training chunks
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
#  RNN Model
# ------------------------------------------------------------------
class SimpleRNN(nn.Module):
    """
    Architecture:
        char index
            -> Embedding (Encoder)
            ->  RNN
            -> Linear (Decoder)
    """

    def __init__(self, input_size, hidden_size, output_size, n_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(
            hidden_size,
            hidden_size,
            n_layers,
            nonlinearity="tanh"
        )
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        """
        x: character index (scalar)
        hidden: (n_layers,1,hidden_size)
        """

        # x.shape: []
        x = x.view(1, -1)              # (1,1)

        x = self.encoder(x)            # (1,1,hidden_size)

        output, hidden = self.rnn(x, hidden)
        # output.shape = (1,1,hidden_size)
        # hidden.shape = (n_layers,1,hidden_size)

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

    # --------------------------------------------------
    # Parsing des arguments (configuration du script)
    # --------------------------------------------------
    parser = ArgumentParser()
    parser.add_argument("--trainingData", default="data/shakespeare.txt")
    parser.add_argument("--trainEval", default="train", choices=["train", "eval"])
    parser.add_argument("--model_dir", default="models/rnn")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=8000)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--length", type=int, default=300)

    args = parser.parse_args()

    # --------------------------------------------------
    # Chargement et normalisation du corpus texte
    # --------------------------------------------------
    text = unidecode.unidecode(open(args.trainingData, encoding="utf-8").read())
    print("Corpus size:", len(text))

    # --------------------------------------------------
    # Initialisation du modèle RNN
    # --------------------------------------------------
    model = SimpleRNN(
        n_characters,              # taille du vocabulaire
        args.hidden_size,           # taille de l'état caché
        n_characters,              # sortie = vocabulaire
        args.num_layers             # nombre de couches
    ).to(device)

    # Optimiseur (Adam) et fonction de perte (CrossEntropy)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Création du dossier de sauvegarde si nécessaire
    if not path.exists(args.model_dir):
        makedirs(args.model_dir)

    model_name = f"rnn_{args.num_layers}_{args.hidden_size}.pt"

    # ==================================================
    # MODE ENTRAÎNEMENT
    # ==================================================
    if args.trainEval == "train":
        start = time.time()

        # Listes pour stocker l'évolution de la loss
        epochs_list = []
        losses_list = []

        for epoch in range(1, args.max_epochs + 1):

            # Sélection aléatoire d'un chunk du texte
            inp, target = random_training_set(text)

            # Une étape d'entraînement (forward + backward)
            loss = train_step(model, inp, target, optimizer, criterion, args.clip)

            # Sauvegarde des valeurs pour l'analyse
            epochs_list.append(epoch)
            losses_list.append(loss)

            # Affichage périodique de l'avancement
            if epoch % 500 == 0:
                print(f"[{epoch}/{args.max_epochs}] loss={loss:.4f} time={time_since(start)}")

        # --------------------------------------------------
        # Pandas : stockage des pertes dans un fichier CSV
        # --------------------------------------------------
        df_loss = pd.DataFrame({
            "epoch": epochs_list,
            "loss": losses_list
        })
        df_loss.to_csv(join(args.model_dir, "training_loss.csv"), index=False)

        # --------------------------------------------------
        # Matplotlib : visualisation de la convergence
        # --------------------------------------------------
        plt.figure(figsize=(8, 5))
        plt.plot(df_loss["epoch"], df_loss["loss"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Evolution (Simple RNN)")
        plt.grid(True)
        plt.savefig(join(args.model_dir, "training_loss.png"))
        plt.close()

        # Sauvegarde du modèle entraîné
        torch.save(model, join(args.model_dir, model_name))
        print("Model saved:", join(args.model_dir, model_name))

    # ==================================================
    # MODE ÉVALUATION / GÉNÉRATION
    # ==================================================
    else:
        # Chargement du modèle entraîné
        model = torch.load(
            join(args.model_dir, model_name),
            map_location=device,
            weights_only=False
        )
        model.eval()  # mode évaluation (pas de gradient)

        # Texte de départ fourni par l'utilisateur
        start_text = input("Enter starting text: ").strip()
        if len(start_text) == 0:
            start_text = "Th"

        # Génération de texte caractère par caractère
        print("\nGenerated text:\n")
        print(evaluate(model, start_text, args.length))
