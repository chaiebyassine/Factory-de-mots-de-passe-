"""
================================================================================
PARTIE 1 - Génération de texte avec LSTM + Dropout
================================================================================
USAGE:
    Entraînement: python partie1_original.py --trainEval train --dropout 0.3
    Évaluation:   python partie1_original.py --trainEval eval --dropout 0.3
================================================================================
"""

import unidecode
import string
import random
import re

from os import listdir, path, makedirs, popen
from os.path import isdir, isfile, join

import torch
import torch.nn as nn
from torch.autograd import Variable

import time, math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from argparse import ArgumentParser

# Vérification GPU/CPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('CUDA AVAILABLE')
else:
    device = torch.device("cpu")
    print('ONLY CPU AVAILABLE')

# Tous les caractères imprimables
all_characters = string.printable
n_characters = len(all_characters)

# Paramètres par défaut
chunk_len = 13
n_epochs = 200000
print_every = 10
plot_every = 10
hidden_size = 512
n_layers = 3
lr = 0.005


def random_chunk(file):
    """Extrait un morceau aléatoire du texte"""
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return file[start_index:end_index]


def char_tensor(string):
    """Convertit une chaîne de caractères en tenseur"""
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor)


def random_training_set(file):
    """Génère un couple (entrée, cible) pour l'entraînement"""
    chunk = random_chunk(file)
    inp = char_tensor(chunk[:-1]).to(device)
    target = char_tensor(chunk[1:]).to(device)
    return inp, target


def evaluate(decoder, prime_str='A', predict_len=100, temperature=0.8):
    """Génère du texte à partir d'une chaîne d'amorçage"""
    hidden = decoder.init_hidden()
    prime_input = char_tensor(prime_str).to(device)
    predicted = prime_str

    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[p], hidden)
    inp = prime_input[-1]

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char).to(device)

    return predicted


def time_since(since):
    """Calcule le temps écoulé depuis 'since'"""
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train(inp, target, clip_grad=0.0):
    """Une étape d'entraînement"""
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0
    
    for c in range(inp.size(0)):
        output, hidden = decoder(inp[c], hidden)
        loss += criterion(output, target[c].unsqueeze(0))

    loss.backward()
    
    # Gradient Clipping - évite l'explosion des gradients
    if clip_grad > 0:
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip_grad)
    
    decoder_optimizer.step()

    return loss.item() / chunk_len


class LSTM(nn.Module):
    """
    LSTM avec Dropout pour la génération de texte
    
    Architecture:
    - Embedding: Encode les caractères en vecteurs denses
    - LSTM: Couche récurrente avec dropout entre couches
    - Dropout: Après la sortie LSTM (avant le décodeur)
    - Linear: Décode vers les probabilités des caractères
    """
    
    def __init__(self, input_size, hidden_size, output_size, n_layers=2, dropout=0.0):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_rate = dropout

        # Couche d'embedding
        self.encoder = nn.Embedding(input_size, hidden_size)
        
        # LSTM avec dropout entre couches (si n_layers > 1)
        rnn_dropout = dropout if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=rnn_dropout)
        
        # Dropout après la sortie LSTM (avant le décodeur)
        self.dropout = nn.Dropout(dropout)
        
        # Couche de décodage
        self.decoder = nn.Linear(hidden_size, output_size)
        
        if dropout > 0:
            print(f"[DROPOUT] Taux: {dropout*100:.0f}% - Appliqué entre couches + avant décodeur")

    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.lstm(input.view(1, 1, -1), hidden)
        output = self.dropout(output)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return (Variable(torch.zeros(self.n_layers, 1, self.hidden_size, device=device)),
                Variable(torch.zeros(self.n_layers, 1, self.hidden_size, device=device)))


def training(n_epochs, file, clip_grad=0.0):
    """Boucle d'entraînement principale"""
    print()
    print('-----------')
    print('|  TRAIN  |')
    print('-----------')
    print()
    
    if clip_grad > 0:
        print(f'[GRADIENT CLIPPING] max_norm = {clip_grad}')

    start = time.time()
    all_losses = []
    loss_avg = 0
    best_loss = 100
    print_every = n_epochs / 100

    for epoch in range(1, n_epochs + 1):
        loss = train(*random_training_set(file), clip_grad=clip_grad)
        loss_avg += loss

        if epoch % print_every == 0:
            print('[%s (%d %d%%) %.4f (%.4f)]' % (time_since(start), epoch, epoch / n_epochs * 100, loss_avg / epoch, loss))

        if best_loss > (loss_avg / epoch):
            best_loss = loss_avg / epoch
            print('[%s (%d %d%%) %.4f (%.4f)]' % (time_since(start), epoch, epoch / n_epochs * 100, loss_avg / epoch, loss))


def evaluating(decoder, length):
    """Mode d'évaluation interactif"""
    print()
    print('------------')
    print('|   EVAL   |')
    print('------------')
    print()

    try:
        while True:
            print('Enter a starting two or tree charachters')
            input1 = input()
            print()
            if len(input1) > 0:
                print('Generated ', length, 'charcaters: ')
                print(evaluate(decoder=decoder, prime_str=input1, predict_len=length, temperature=0.8))
            else:
                print(input1, ' length < 1')
            print('------------')
            print()

    except KeyboardInterrupt:
        print("Press Ctrl-C to terminate evaluating")
        print('------------')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-d", "--trainingData", default="data/shakespeare.txt", type=str,
                        help="trainingData [path/to/the/data]")
    parser.add_argument("-te", "--trainEval", default='train', type=str,
                        help="trainEval [train, eval]")
    parser.add_argument("-m", "--model", default='models/lstm', type=str,
                        help="model to save (train) or to load (eval) [path/to/the/model]")
    parser.add_argument('--length', default=100, type=int,
                        help="sequence length during eval process [< 1000]")
    parser.add_argument('--clip', default=0.0, type=float,
                        help="gradient clipping (0.0 = désactivé, recommandé: 1.0-5.0)")
    parser.add_argument('--num_layers', default=2, type=int,
                        help="nombre de couches LSTM")
    parser.add_argument('--hidden_size', default=256, type=int,
                        help="taille de la couche cachée")
    parser.add_argument('--max_epochs', default=5000, type=int,
                        help="nombre d'époques d'entraînement")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="taux de dropout (0.0 à 0.5)")

    args = parser.parse_args()

    # Chargement des données
    repData = args.trainingData
    file = unidecode.unidecode(open(repData, encoding='utf-8').read())
    file_len = len(file)

    # Création du modèle LSTM
    decoder = LSTM(n_characters, args.hidden_size, n_characters, args.num_layers, args.dropout).to(device)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n_epochs = args.max_epochs

    # Affichage d'un échantillon
    print()
    print("=" * 60)
    print("Génération de texte avec LSTM")
    print("=" * 60)
    print()
    print('Échantillon du texte:')
    print(random_chunk(file))
    print()
    print('Taille du fichier:', file_len, 'caractères')
    print(f'Modèle: LSTM avec {args.num_layers} couches, {args.hidden_size} unités cachées')
    if args.dropout > 0:
        print(f'Dropout: {args.dropout*100:.0f}%')
    print()

    # Nom du fichier modèle
    if args.dropout > 0:
        dropout_str = str(int(args.dropout * 100))
        modelFile = f"lstmGeneration_{args.num_layers}_{args.hidden_size}_dropout{dropout_str}.pt"
    else:
        modelFile = f"lstmGeneration_{args.num_layers}_{args.hidden_size}.pt"

    # Créer le répertoire models si nécessaire
    if not path.exists(args.model):
        makedirs(args.model)

    # Mode entraînement ou évaluation
    if args.trainEval == 'train':
        decoder.train()
        training(n_epochs, file, clip_grad=args.clip)
        torch.save(decoder, join(args.model, modelFile))
        print()
        print('Modèle sauvegardé:', join(args.model, modelFile))
    elif args.trainEval == 'eval':
        decoder = torch.load(join(args.model, modelFile))
        decoder.eval().to(device)
        evaluating(decoder, args.length)
    else:
        print('Choose trainEval option (--trainEval train/eval')
