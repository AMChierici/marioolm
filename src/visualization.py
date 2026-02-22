"""
Visualization module for Italian Lyrics ML.

Visualizations make the invisible visible. When an AI is training,
all that's really happening is numbers changing — but plots and charts
help us see the patterns and understand what's going on.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use a clean style for all plots
matplotlib.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def plot_training_loss(histories, title="Training Loss Over Time"):
    """
    Plot how the loss decreases during training for each model.

    What you're seeing:
    - X-axis: epochs (number of times the model has read all the data)
    - Y-axis: loss (how wrong the model is — lower is better)
    - Each line is a different model architecture

    A steep drop means the model is learning quickly.
    A flat line means the model has stopped improving.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'RNN': '#e74c3c', 'LSTM': '#3498db', 'Transformer': '#2ecc71'}

    for history in histories:
        name = history['model_name']
        losses = history['epoch_losses']
        epochs = range(1, len(losses) + 1)
        color = colors.get(name, '#95a5a6')
        ax.plot(epochs, losses, 'o-', label=name, color=color,
                linewidth=2, markersize=8)

    ax.set_xlabel('Epoch (full pass through all songs)')
    ax.set_ylabel('Loss (how wrong the model is — lower is better)')
    ax.set_title(title)
    ax.legend(fontsize=12)
    plt.tight_layout()
    return fig


def plot_training_time_comparison(histories):
    """
    Bar chart comparing how long each model took to train.

    This shows a key real-world trade-off:
    - Simple models (RNN) train fast but produce lower quality
    - Complex models (Transformer) take much longer but produce better results
    - The same trade-off exists at massive scale: training GPT-4 costs
      millions of dollars in compute time
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    names = [h['model_name'] for h in histories]
    times = [h['training_time'] / 60 for h in histories]  # Convert to minutes
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    bars = ax.bar(names, times, color=colors[:len(names)], edgecolor='white',
                  linewidth=2)

    # Add time labels on bars
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{t:.1f} min', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Training Time (minutes)')
    ax.set_title('Training Time Comparison\n'
                 'More powerful models take longer to train')
    plt.tight_layout()
    return fig


def plot_parameter_comparison(param_counts):
    """
    Bar chart comparing the number of parameters (learnable numbers)
    in each model.

    Parameters are the individual numbers that get adjusted during training.
    More parameters = more capacity to learn patterns, but also:
    - More memory needed
    - Longer training time
    - Risk of "overfitting" (memorizing instead of learning)
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(param_counts.keys())
    counts = [v / 1_000_000 for v in param_counts.values()]  # Millions
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    bars = ax.bar(names, counts, color=colors[:len(names)], edgecolor='white',
                  linewidth=2)

    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{c:.1f}M', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Parameters (millions)')
    ax.set_title('Model Size Comparison\n'
                 'GPT-2 has ~100x more parameters than RNN/LSTM')
    plt.tight_layout()
    return fig


def visualize_tokenization(tokenizer, text, max_tokens=30):
    """
    Show how a tokenizer breaks text into tokens with color coding.

    This is one of the most important concepts in AI:
    Before an AI can read text, it must convert words into numbers.
    The tokenizer is the "dictionary" that maps between the two.

    Some words map to one token, others get split into pieces.
    For example: "incredible" might become ["incred", "ible"]
    """
    tokens = tokenizer.encode(text)[:max_tokens]
    decoded = [tokenizer.decode([t]) for t in tokens]

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.set_xlim(0, len(decoded))
    ax.set_ylim(0, 2)
    ax.axis('off')
    ax.set_title(f'Tokenization: how AI reads "{text[:50]}..."'
                 if len(text) > 50 else f'Tokenization: how AI reads "{text}"',
                 fontsize=14, pad=20)

    cmap = plt.cm.Set3
    for i, (token_text, token_id) in enumerate(zip(decoded, tokens)):
        color = cmap(i % 12)
        # Token text
        ax.add_patch(plt.Rectangle((i, 1.0), 0.95, 0.8,
                                   facecolor=color, edgecolor='gray'))
        display_text = repr(token_text) if token_text.strip() == '' else token_text
        ax.text(i + 0.47, 1.4, display_text, ha='center', va='center',
                fontsize=9, fontweight='bold')
        # Token ID
        ax.add_patch(plt.Rectangle((i, 0.1), 0.95, 0.7,
                                   facecolor='white', edgecolor='gray'))
        ax.text(i + 0.47, 0.45, str(token_id), ha='center', va='center',
                fontsize=8, color='#666')

    # Labels
    ax.text(-0.3, 1.4, 'Text:', ha='right', va='center',
            fontsize=10, fontweight='bold')
    ax.text(-0.3, 0.45, 'Token ID:', ha='right', va='center',
            fontsize=10, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_next_word_probabilities(probabilities, words, title="What word comes next?"):
    """
    Horizontal bar chart showing the model's predicted probabilities
    for the next word.

    This reveals how the model "thinks":
    - High confidence in one word = the model is very sure
    - Spread across many words = the model sees multiple possibilities
    - This is exactly what happens inside ChatGPT billions of times per response
    """
    fig, ax = plt.subplots(figsize=(10, max(4, len(words) * 0.5)))

    y_pos = np.arange(len(words))
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(words)))

    bars = ax.barh(y_pos, probabilities, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words, fontsize=11)
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, max(probabilities) * 1.15)

    for bar, prob in zip(bars, probabilities):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{prob:.1%}', va='center', fontsize=10)

    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def display_generation_comparison(results, title="Model Output Comparison"):
    """
    Print a formatted comparison of text generated by different models.

    Args:
        results: Dict of {model_name: generated_text}
    """
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    for name, text in results.items():
        print(f"\n  [{name}]")
        # Wrap long lines for readability
        words = text.split()
        line = "  "
        for word in words:
            if len(line) + len(word) + 1 > 58:
                print(line)
                line = "  " + word
            else:
                line += " " + word if line.strip() else "  " + word
        print(line)
    print(f"\n{'=' * 60}")
