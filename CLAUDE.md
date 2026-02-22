# CLAUDE.md

## Project Overview

Italian Lyrics ML — a machine learning project that trains neural network models (RNN, LSTM, GPT-2 Transformer) on Italian music lyrics for text generation. Built with PyTorch and Hugging Face Transformers.

## Repository Structure

```
marioolm/
├── main.py                        # Primary training & inference script (models, training loop, generation)
├── json_to_lyrics_converter.py    # Preprocesses raw JSON dataset into plain text lyrics
├── check_environment.py           # Validates Python environment and dependencies
├── Untitled.ipynb                 # Jupyter notebook with experiments and improved generation
├── environment.yml                # Conda environment (minimal spec)
├── environment_full.yml           # Conda environment (pinned versions, 211 packages)
├── .git_ignore                    # Git ignore rules
└── data/
    ├── the_italian_music_dataset.json        # Full dataset (~14,679 songs, JSONL)
    ├── the_italian_music_dataset_sample.json  # Sample dataset
    ├── the_Italian_Music_dataset_v1.zip       # Compressed dataset
    └── italian_lyrics.txt                     # Processed Italian-only lyrics (~9,135 lines)
```

## Key Components

### `main.py`
- `ItalianLyricsDataset` — PyTorch `Dataset` subclass; tokenizes lyrics with GPT-2 tokenizer, pads/truncates to `max_length`
- `RNNModel` — Embedding → RNN → Linear; forward pass takes input_ids only (no attention_mask)
- `LSTMModel` — Embedding → LSTM → Linear; same interface as RNNModel
- `train_model()` — Generic training loop for all model types; uses CrossEntropyLoss with next-token prediction (input shifted against itself)
- `generate_lyrics()` — Text generation using `model.generate()` (works for Transformer; RNN/LSTM lack `.generate()`)
- `main()` — Orchestrates loading data, training all three models, and generating samples

### `json_to_lyrics_converter.py`
- Reads JSONL dataset, extracts `lyrics` field, filters to Italian using `langdetect`
- Outputs one song per line (newlines within lyrics replaced with spaces)
- Run as a standalone script: `python json_to_lyrics_converter.py`

### `check_environment.py`
- Verifies Python version, imports of key libraries, and CUDA availability
- Run as: `python check_environment.py`

### `Untitled.ipynb`
- Contains experimental code with improvements over `main.py`:
  - Top-k and top-p (nucleus) sampling for better generation quality
  - Gradient clipping to prevent exploding gradients
  - Per-model learning rate adjustments
  - Custom generation functions for RNN/LSTM models

## Development Setup

### Environment

Uses Conda with Python 3.9:

```bash
conda env create -f environment.yml
conda activate italian-lyrics-ml
```

For exact version reproduction:
```bash
conda env create -f environment_full.yml
```

### Key Dependencies

- **PyTorch 1.10** — model training and inference
- **Transformers (Hugging Face)** — GPT-2 tokenizer and pretrained model
- **langdetect** — Italian language filtering during preprocessing
- **pandas, numpy** — data handling
- **scikit-learn** — utilities
- **matplotlib** — plotting
- **tqdm** — progress bars
- **JupyterLab** — notebook execution

### Validate Setup

```bash
python check_environment.py
```

## Data Pipeline

1. Raw data: `data/the_italian_music_dataset.json` (JSONL, ~14,679 songs with metadata)
2. Run `python json_to_lyrics_converter.py` to filter Italian lyrics → `data/italian_lyrics.txt`
3. `main.py` reads `data/italian_lyrics.txt` for training

## Model Hyperparameters

| Parameter       | Value |
|-----------------|-------|
| Embedding dim   | 256   |
| Hidden dim      | 512   |
| Max seq length  | 128   |
| Batch size      | 32    |
| Epochs          | 5     |
| Optimizer       | Adam  |
| Tokenizer       | GPT-2 (50257 vocab) |

## Known Issues and Gotchas

- **RNN/LSTM models lack `.generate()`** — The `generate_lyrics()` function in `main.py` calls `model.generate()`, which only exists on Hugging Face models (GPT-2). RNN and LSTM need custom generation loops (see notebook for working implementations).
- **RNN/LSTM don't use attention_mask** — The `train_model()` function unpacks `(inputs, masks)` from the dataloader but only passes `inputs` to RNN/LSTM models. The mask is loaded but unused for these models.
- **Repetition collapse** — RNN/LSTM models tend to generate repetitive text without sampling strategies. The notebook implements top-k/top-p sampling to address this.
- **Transformer training is slow** — GPT-2 fine-tuning is ~100x slower than RNN/LSTM training on CPU.
- **Data files are gitignored** — The `data/` directory is excluded from version control via `.git_ignore`. You need the dataset files present locally to run training.

## Conventions

- **No formal test suite** — There are no pytest/unittest tests. `check_environment.py` is the only validation script.
- **No linter or formatter configured** — No flake8, black, ruff, or similar tools are set up.
- **No CI/CD pipeline** — No GitHub Actions or other CI configuration.
- **Single-file architecture** — All model definitions, training, and inference live in `main.py`. Experimental improvements are in the notebook.
- **Git ignore file** — Named `.git_ignore` (not `.gitignore`). Contains rules for `/data*` and `/.ipynb_checkpoints*`.

## Running the Project

```bash
# 1. Set up environment
conda env create -f environment.yml
conda activate italian-lyrics-ml

# 2. Validate environment
python check_environment.py

# 3. Preprocess data (if italian_lyrics.txt doesn't exist)
python json_to_lyrics_converter.py

# 4. Train models and generate lyrics
python main.py
```

## File Encoding

All Python files and data files use UTF-8 encoding. This is important given the Italian-language content with accented characters (à, è, é, ì, ò, ù).
