"""
Model definitions for Italian Lyrics ML.

This module defines three increasingly powerful neural network architectures
for generating Italian song lyrics. Each model learns to predict the next
word in a sequence — the same fundamental idea behind ChatGPT.

In plain language:
- RNN:         Like reading a sentence and only remembering the last few words.
- LSTM:        Like reading a sentence and remembering the important parts.
- Transformer: Like reading the whole sentence at once and understanding
               how every word relates to every other word.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel


class RNNModel(nn.Module):
    """
    Recurrent Neural Network — the simplest architecture.

    How it works:
    1. Each word is converted to a vector of numbers (embedding)
    2. The network reads words one by one, left to right
    3. At each step it updates a "hidden state" — its memory
    4. Problem: it tends to forget earlier words in long sequences

    Architecture: Embedding -> RNN -> Linear output
    """

    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        return self.fc(output)


class LSTMModel(nn.Module):
    """
    Long Short-Term Memory network — improved memory over RNN.

    How it works:
    1. Same word-by-word reading as RNN
    2. But adds "gates" that control what to remember and what to forget
    3. Can retain important information over longer sequences
    4. Still reads sequentially — can't look ahead

    Architecture: Embedding -> LSTM -> Linear output
    """

    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return self.fc(output)


def create_transformer_model(tokenizer):
    """
    Load a pre-trained GPT-2 Transformer model.

    Unlike RNN/LSTM which we build from scratch, the Transformer starts
    with knowledge from reading millions of English web pages. We then
    "fine-tune" it on Italian lyrics — teaching an English reader to
    write Italian songs.

    This is called "transfer learning" — reusing knowledge from one
    task to help with another. It's why modern AI is so powerful.
    """
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))
    return model


def count_parameters(model):
    """Count the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model, name="Model"):
    """Print a human-readable summary of a model's size."""
    total = count_parameters(model)
    if total >= 1_000_000:
        print(f"{name}: {total / 1_000_000:.1f} million parameters")
    else:
        print(f"{name}: {total:,} parameters")
    return total
