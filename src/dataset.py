"""
Dataset module for Italian Lyrics ML.

This module handles loading Italian song lyrics and preparing them
for training. Think of it as the "reading" step — before an AI can
learn from text, that text needs to be converted into numbers.
"""

import torch
from torch.utils.data import Dataset


class ItalianLyricsDataset(Dataset):
    """
    A PyTorch Dataset that converts Italian lyrics into sequences of numbers
    that a neural network can process.

    How it works (in plain language):
    1. Takes a list of song lyrics (plain text strings)
    2. Uses a "tokenizer" to convert each word/subword into a number
    3. Pads short lyrics or trims long ones so they're all the same length
    4. Returns the number sequences ready for training
    """

    def __init__(self, lyrics, tokenizer, max_length=128):
        self.lyrics = lyrics
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.lyrics)

    def __getitem__(self, idx):
        lyric = self.lyrics[idx]
        encoding = self.tokenizer(
            lyric,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()


def load_lyrics(filepath, max_songs=None):
    """
    Load lyrics from a text file (one song per line).

    Args:
        filepath: Path to the lyrics text file.
        max_songs: Optional limit on number of songs to load.
                   Useful for quick experiments.

    Returns:
        List of lyric strings.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lyrics = f.readlines()

    if max_songs is not None:
        lyrics = lyrics[:max_songs]

    return lyrics
