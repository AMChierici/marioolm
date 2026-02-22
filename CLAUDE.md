# CLAUDE.md

## Project Overview

Italian Lyrics ML — an **educational project** that teaches non-technical people how AI works by training neural network models (RNN, LSTM, GPT-2 Transformer) on Italian music lyrics for text generation. Built with PyTorch and Hugging Face Transformers.

The primary audience is **non-technical learners**. All code, notebooks, and documentation should prioritize clarity and plain-language explanations over technical brevity.

## Repository Structure

```
marioolm/
├── README.md                      # Project introduction and learning path
├── CLAUDE.md                      # AI assistant context (this file)
├── main.py                        # Full pipeline script (train all models + generate)
│
├── notebooks/                     # Guided learning path (7 notebooks)
│   ├── 01_exploring_the_data.ipynb          # What is training data?
│   ├── 02_how_ai_reads_text.ipynb           # Tokenization explained
│   ├── 03_training_a_simple_rnn.ipynb       # First model, watching it learn
│   ├── 04_lstm_better_memory.ipynb          # LSTM vs RNN comparison
│   ├── 05_transformer_revolution.ipynb      # GPT-2, transfer learning, scale
│   ├── 06_improving_generation.ipynb        # Sampling strategies, fixing repetition
│   └── 07_limitations_ethics_opportunities.ipynb  # Bias, cost, real-world impact
│
├── src/                           # Source code modules (used by notebooks and main.py)
│   ├── __init__.py
│   ├── dataset.py                 # ItalianLyricsDataset, load_lyrics()
│   ├── models.py                  # RNNModel, LSTMModel, create_transformer_model()
│   ├── training.py                # train_model(), get_optimizer()
│   ├── generation.py              # generate_rnn_lstm(), generate_transformer(), generate_greedy()
│   └── visualization.py           # Plotting: loss curves, tokenization, probabilities
│
├── scripts/                       # Utility scripts
│   ├── preprocess_data.py         # Convert raw JSON dataset to training text
│   └── check_environment.py       # Verify Python environment and dependencies
│
├── json_to_lyrics_converter.py    # Original preprocessing script (kept for reference)
├── check_environment.py           # Original environment check (kept for reference)
├── Untitled.ipynb                 # Original experimental notebook (kept for reference)
│
├── data/                          # Dataset (gitignored — not in version control)
│   ├── the_italian_music_dataset.json        # Full dataset (~14,679 songs, JSONL)
│   ├── the_italian_music_dataset_sample.json  # Sample dataset
│   ├── the_Italian_Music_dataset_v1.zip       # Compressed dataset
│   └── italian_lyrics.txt                     # Processed Italian-only lyrics (~9,135 songs)
│
├── environment.yml                # Conda environment (minimal spec)
├── environment_full.yml           # Conda environment (pinned versions)
└── .gitignore                     # Git ignore rules
```

## Key Source Modules (`src/`)

### `src/dataset.py`
- `ItalianLyricsDataset` — PyTorch Dataset; tokenizes lyrics with GPT-2 tokenizer, pads/truncates to max_length
- `load_lyrics(filepath, max_songs=None)` — Load lyrics from text file with optional limit

### `src/models.py`
- `RNNModel` — Embedding → RNN → Linear
- `LSTMModel` — Embedding → LSTM → Linear
- `create_transformer_model(tokenizer)` — Load pretrained GPT-2 with resized embeddings
- `model_summary(model, name)` — Print human-readable parameter count
- `count_parameters(model)` — Return total trainable parameters

### `src/training.py`
- `train_model()` — Generic training loop; handles both Transformer and RNN/LSTM; includes gradient clipping; returns training history dict
- `get_optimizer(model, is_transformer)` — Returns Adam optimizer with appropriate learning rate (5e-5 for Transformer, 1e-3 for RNN/LSTM)

### `src/generation.py`
- `generate_rnn_lstm()` — Custom generation loop with top-k/top-p sampling for RNN and LSTM models
- `generate_transformer()` — GPT-2 generation with sampling and repetition penalty
- `generate_greedy()` — Intentionally simple greedy decoding (for educational comparison)
- `top_k_top_p_filtering()` — Nucleus sampling implementation

### `src/visualization.py`
- `plot_training_loss(histories)` — Multi-model loss curves
- `plot_training_time_comparison(histories)` — Bar chart of training times
- `plot_parameter_comparison(param_counts)` — Bar chart of model sizes
- `visualize_tokenization(tokenizer, text)` — Color-coded token visualization
- `plot_next_word_probabilities(probs, words)` — Horizontal bar chart of predictions
- `display_generation_comparison(results)` — Formatted text comparison

## Educational Design Principles

When modifying this project, follow these principles:

1. **Plain language first** — Every code cell should be preceded by a markdown cell explaining the concept in everyday terms. Use analogies.
2. **Show, don't just tell** — Use visualizations wherever possible. A chart of training loss is more meaningful than printing numbers.
3. **Embrace failure as teaching** — Repetition collapse, slow training, and incoherent output are FEATURES in this context. They demonstrate real AI limitations.
4. **Build incrementally** — Each notebook builds on the previous. Concepts introduced in notebook 02 are used without re-explanation in notebook 05.
5. **Connect to the real world** — Always bridge from the small experiment to real AI systems (ChatGPT, etc.).

## Development Setup

### Environment

```bash
conda env create -f environment.yml
conda activate italian-lyrics-ml
```

### Key Dependencies

- **PyTorch 1.10** — model training and inference
- **Transformers (Hugging Face)** — GPT-2 tokenizer and pretrained model
- **langdetect** — Italian language filtering during preprocessing
- **pandas, numpy** — data handling
- **matplotlib** — plotting and visualization
- **scikit-learn** — utilities
- **tqdm** — progress bars
- **JupyterLab** — notebook execution

### Validate Setup

```bash
python scripts/check_environment.py
```

## Data Pipeline

1. Raw data: `data/the_italian_music_dataset.json` (JSONL, ~14,679 songs with metadata)
2. Run `python scripts/preprocess_data.py` to filter Italian lyrics → `data/italian_lyrics.txt`
3. Notebooks and `main.py` read `data/italian_lyrics.txt` for training

## Model Hyperparameters

| Parameter       | Value |
|-----------------|-------|
| Embedding dim   | 256   |
| Hidden dim      | 512   |
| Max seq length  | 128   |
| Batch size      | 32    |
| Epochs          | 3 (notebooks) / 5 (original) |
| Learning rate   | 1e-3 (RNN/LSTM), 5e-5 (Transformer) |
| Optimizer       | Adam  |
| Tokenizer       | GPT-2 (50,257 vocab) |
| Gradient clip   | max_norm=1.0 |

## Known Issues and Gotchas

- **RNN/LSTM don't use attention_mask** — The training loop unpacks `(inputs, masks)` from the dataloader but only passes `inputs` to RNN/LSTM. The mask is used only for Transformer training.
- **Repetition collapse** — Greedy decoding produces repetitive text. The `generate_greedy()` function is intentionally included to demonstrate this problem; `generate_rnn_lstm()` and `generate_transformer()` use sampling to fix it.
- **Transformer training is slow** — GPT-2 fine-tuning is ~100x slower than RNN/LSTM on CPU.
- **English tokenizer on Italian text** — GPT-2's tokenizer was trained on English; Italian words may be split into more subword tokens than necessary, reducing efficiency.
- **Data files are gitignored** — The `data/` directory is excluded. Dataset files must be present locally.
- **Notebooks use relative imports** — All notebooks use `sys.path.insert(0, '..')` to import from `src/`. They must be run from the `notebooks/` directory or via `jupyter lab notebooks/`.

## Conventions

- **No formal test suite** — No pytest/unittest. `scripts/check_environment.py` is the only validation.
- **No linter or formatter** — No flake8, black, ruff, or similar configured.
- **No CI/CD pipeline** — No GitHub Actions or CI configuration.
- **Modular architecture** — Source code in `src/`, educational content in `notebooks/`, utilities in `scripts/`.
- **Docstrings are educational** — Module and function docstrings include plain-language explanations, not just technical API docs.
- **UTF-8 encoding** — All files use UTF-8 for Italian characters (à, è, é, ì, ò, ù).

## Running the Project

```bash
# 1. Set up environment
conda env create -f environment.yml
conda activate italian-lyrics-ml

# 2. Validate environment
python scripts/check_environment.py

# 3. Preprocess data (if italian_lyrics.txt doesn't exist)
python scripts/preprocess_data.py

# 4. Start the guided learning path
jupyter lab notebooks/

# Or run the full pipeline at once
python main.py
```
