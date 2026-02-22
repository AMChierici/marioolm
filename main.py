"""
Italian Lyrics ML — Main training and generation script.

This script trains three neural network models on Italian song lyrics
and generates sample text from each. It serves as a quick way to run
the full pipeline. For a guided educational experience, see the
notebooks in the notebooks/ directory.

Usage:
    python main.py
"""

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from src.dataset import ItalianLyricsDataset, load_lyrics
from src.models import RNNModel, LSTMModel, create_transformer_model, model_summary
from src.training import train_model, get_optimizer
from src.generation import generate_rnn_lstm, generate_transformer
from src.visualization import (
    plot_training_loss,
    plot_training_time_comparison,
    display_generation_comparison,
)


def main():
    # --- Configuration ---
    MAX_SONGS = 1000       # Use fewer songs for faster training (None = all)
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    MAX_LENGTH = 128
    BATCH_SIZE = 32
    EPOCHS = 3
    SEED_TEXT = "Amore mio"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # --- Load Data ---
    print("Loading lyrics...")
    lyrics = load_lyrics('./data/italian_lyrics.txt', max_songs=MAX_SONGS)
    print(f"Loaded {len(lyrics)} songs\n")

    # --- Initialize Tokenizer ---
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # --- Create Dataset & DataLoader ---
    dataset = ItalianLyricsDataset(lyrics, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Initialize Models ---
    vocab_size = tokenizer.vocab_size

    rnn_model = RNNModel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM).to(device)
    lstm_model = LSTMModel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM).to(device)
    transformer_model = create_transformer_model(tokenizer).to(device)

    print("Model sizes:")
    model_summary(rnn_model, "RNN")
    model_summary(lstm_model, "LSTM")
    model_summary(transformer_model, "Transformer (GPT-2)")
    print()

    # --- Train All Models ---
    histories = []

    models_config = [
        ("RNN", rnn_model, False),
        ("LSTM", lstm_model, False),
        ("Transformer", transformer_model, True),
    ]

    for name, model, is_transformer in models_config:
        print(f"Training {name}...")
        optimizer = get_optimizer(model, is_transformer=is_transformer)
        history = train_model(
            model, dataloader, optimizer, device, EPOCHS,
            is_transformer=is_transformer, model_name=name
        )
        histories.append(history)
        print()

    # --- Visualize Training ---
    print("Generating training plots...")
    fig1 = plot_training_loss(histories)
    fig1.savefig('training_loss.png', dpi=100, bbox_inches='tight')
    fig2 = plot_training_time_comparison(histories)
    fig2.savefig('training_time.png', dpi=100, bbox_inches='tight')
    print("Saved: training_loss.png, training_time.png\n")

    # --- Generate Lyrics ---
    print(f"Generating lyrics from seed: \"{SEED_TEXT}\"\n")
    results = {
        'RNN': generate_rnn_lstm(rnn_model, tokenizer, SEED_TEXT),
        'LSTM': generate_rnn_lstm(lstm_model, tokenizer, SEED_TEXT),
        'Transformer': generate_transformer(transformer_model, tokenizer, SEED_TEXT),
    }

    display_generation_comparison(results)


if __name__ == "__main__":
    main()
