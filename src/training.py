"""
Training module for Italian Lyrics ML.

Training is how an AI model "learns." The process is simple in concept:
1. Show the model some text
2. Ask it to predict the next word
3. Tell it how wrong it was (the "loss")
4. Adjust its internal numbers slightly to be less wrong next time
5. Repeat millions of times

The "loss" number measures how wrong the model is — lower is better.
Watching the loss decrease over epochs (full passes through the data)
is like watching a student gradually improve at a subject.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, dataloader, optimizer, device, epochs,
                is_transformer=False, model_name="Model"):
    """
    Train a model on the lyrics dataset.

    Args:
        model: The neural network to train.
        dataloader: Provides batches of training data.
        optimizer: Controls how the model's numbers are adjusted.
        device: CPU or GPU.
        epochs: How many times to read through all the data.
        is_transformer: Whether this is a Transformer model
                        (they handle loss calculation differently).
        model_name: Display name for progress messages.

    Returns:
        Dictionary with training history (loss per epoch, timing).
    """
    criterion = nn.CrossEntropyLoss()
    model.train()

    history = {
        'epoch_losses': [],
        'training_time': 0,
        'model_name': model_name
    }

    start_time = time.time()

    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0

        for batch in dataloader:
            inputs, masks = batch
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()

            if is_transformer:
                outputs = model(inputs, attention_mask=masks, labels=inputs)
                loss = outputs.loss
            else:
                outputs = model(inputs)
                loss = criterion(
                    outputs.view(-1, outputs.size(-1)),
                    inputs.view(-1)
                )

            loss.backward()
            # Gradient clipping prevents "exploding gradients" — when
            # adjustments become too large and destabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count
        history['epoch_losses'].append(avg_loss)
        print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    history['training_time'] = elapsed
    print(f"  {model_name} training completed in {elapsed / 60:.1f} minutes")

    return history


def get_optimizer(model, is_transformer=False):
    """
    Create an optimizer with appropriate learning rate.

    The learning rate controls how big each adjustment step is:
    - Too high: the model overshoots and never settles on good values
    - Too low: training takes forever
    - Transformers need a much smaller learning rate because they
      already have pre-trained knowledge we don't want to destroy
    """
    if is_transformer:
        lr = 5e-5   # Very small steps — preserve pre-trained knowledge
    else:
        lr = 0.001  # Larger steps — learning from scratch
    return optim.Adam(model.parameters(), lr=lr)
