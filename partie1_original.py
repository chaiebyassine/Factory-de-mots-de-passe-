"""
================================================================================
PARTIE 1 - Code Original du Professeur (Multi-Architecture)
================================================================================
Génération de texte avec RNN, GRU ou LSTM - Version configurable

OBJECTIFS:
- Récupérer et traiter les données ✓
- Apprendre sur ces données avec différentes architectures (RNN, GRU, LSTM)
- Comparer les performances des différents modèles
- Évaluer le système IA lors de la phase de génération de texte

USAGE:
    RNN:  python partie1_original.py --trainEval train --model_type rnn
    GRU:  python partie1_original.py --trainEval train --model_type gru
    LSTM: python partie1_original.py --trainEval train --model_type lstm
    
    Évaluation: python partie1_original.py --trainEval eval --model_type [rnn/gru/lstm]
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
    """
    Génère du texte à partir d'une chaîne d'amorçage
    
    Args:
        decoder: Le modèle RNN
        prime_str: Chaîne de départ pour la génération
        predict_len: Nombre de caractères à générer
        temperature: Contrôle la créativité (plus bas = plus conservateur)
    """
    hidden = decoder.init_hidden()
    prime_input = char_tensor(prime_str).to(device)
    predicted = prime_str

    # Utiliser la chaîne d'amorçage pour construire l'état caché
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[p], hidden)
    inp = prime_input[-1]

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)

        # Échantillonnage avec distribution multinomiale
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Ajouter le caractère prédit et l'utiliser comme prochaine entrée
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


def train(inp, target):
    """Une étape d'entraînement"""
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0
    
    for c in range(inp.size(0)):
        output, hidden = decoder(inp[c], hidden)
        loss += criterion(output, target[c].unsqueeze(0))

    loss.backward()
    decoder_optimizer.step()

    return loss.item() / chunk_len


class RNN(nn.Module):
    """
    Réseau de Neurones Récurrent - Multi-Architecture (RNN, GRU, LSTM)
    
    Architecture:
    - Embedding: Encode les caractères en vecteurs denses
    - Couche récurrente: RNN, GRU ou LSTM selon le choix
    - Linear: Décode vers les probabilités des caractères
    """
    
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, model_type='gru'):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.model_type = model_type.lower()

        # Couche d'embedding
        self.encoder = nn.Embedding(input_size, hidden_size)
        
        # Couche récurrente selon le type choisi
        if self.model_type == 'rnn':
            self.rnn = nn.RNN(hidden_size, hidden_size, n_layers)
        elif self.model_type == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.model_type == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        else:
            raise ValueError(f"Type de modèle non supporté: {model_type}. Utilisez 'rnn', 'gru' ou 'lstm'")
        
        # Couche de décodage
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.rnn(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        # LSTM utilise un tuple (hidden_state, cell_state)
        if self.model_type == 'lstm':
            return (Variable(torch.zeros(self.n_layers, 1, self.hidden_size, device=device)),
                    Variable(torch.zeros(self.n_layers, 1, self.hidden_size, device=device)))
        else:
            return Variable(torch.zeros(self.n_layers, 1, self.hidden_size, device=device))


def training(n_epochs, file):
    """Boucle d'entraînement principale"""
    print()
    print('-----------')
    print('|  TRAIN  |')
    print('-----------')
    print()

    start = time.time()
    all_losses = []
    loss_avg = 0
    best_loss = 100
    print_every = n_epochs / 100

    for epoch in range(1, n_epochs + 1):
        loss = train(*random_training_set(file))
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
    parser.add_argument("-r", "--run", default="rnnGeneration", type=str,
                        help="name of the model saved file")
    parser.add_argument("-m", "--model", default='models', type=str,
                        help="model to save (train) or to load (eval) [path/to/the/model]")
    parser.add_argument('--length', default=100, type=int,
                        help="sequence length during eval process [< 1000]")
    parser.add_argument('--num_layers', default=2, type=int,
                        help="nombre de couches GRU")
    parser.add_argument('--hidden_size', default=128, type=int,
                        help="taille de la couche cachée")
    parser.add_argument('--max_epochs', default=10000, type=int,
                        help="nombre d'époques d'entraînement")
    parser.add_argument('--model_type', default='gru', type=str,
                        choices=['rnn', 'gru', 'lstm'],
                        help="type de modèle: rnn, gru, ou lstm")

    args = parser.parse_args()

    # Chargement des données
    repData = args.trainingData
    file = unidecode.unidecode(open(repData, encoding='utf-8').read())
    file_len = len(file)

    # Type de modèle
    model_type = args.model_type.lower()
    model_type_upper = model_type.upper()

    # Création du modèle avec le type choisi
    decoder = RNN(n_characters, args.hidden_size, n_characters, args.num_layers, model_type).to(device)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n_epochs = args.max_epochs

    # Affichage d'un échantillon
    print()
    print("=" * 60)
    print(f"PARTIE 1 - Génération de texte avec {model_type_upper}")
    print("=" * 60)
    print()
    print('Échantillon du texte:')
    print(random_chunk(file))
    print()
    print('Taille du fichier:', file_len, 'caractères')
    print(f'Modèle: {model_type_upper} avec {args.num_layers} couches, {args.hidden_size} unités cachées')
    print()

    # Nom du fichier modèle (inclut le type de modèle)
    modelFile = f"{model_type}Generation_{args.num_layers}_{args.hidden_size}.pt"

    # Créer le répertoire models si nécessaire
    if not path.exists(args.model):
        makedirs(args.model)

    # Mode entraînement ou évaluation
    if args.trainEval == 'train':
        decoder.train()
        training(n_epochs, file)
        torch.save(decoder, join(args.model, modelFile))
        print()
        print('Modèle sauvegardé:', join(args.model, modelFile))
    elif args.trainEval == 'eval':
        decoder = torch.load(join(args.model, modelFile))
        decoder.eval().to(device)
        evaluating(decoder, args.length)
    else:
        print('Choose trainEval option (--trainEval train/eval')
