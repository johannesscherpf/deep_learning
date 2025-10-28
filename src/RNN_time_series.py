import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
from torch import Tensor


data = pd.read_csv('../data/rnn_train.csv')
print(data.head())

def create_sequences(data):
    sequences = []
    for column in data.columns:
        sequences.append(list(data[column]))
    sequences = np.array(sequences)
    return Tensor(sequences)

sequences = create_sequences(data)

def create_y(df):
    y = list(data.columns)
    return y