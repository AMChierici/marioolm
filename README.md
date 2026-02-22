# Italian Lyrics ML — Learn How AI Works

**An educational project that teaches you how artificial intelligence creates text, by training AI models to write Italian song lyrics.**

No technical background required. If you've ever wondered how ChatGPT, Gemini, or other AI systems actually work under the hood, this project walks you through it step by step — using the beautiful domain of Italian music.

---

## What You'll Learn

By working through this project, you'll understand:

1. **How AI reads text** — Computers can't read words. They need numbers. You'll see exactly how text gets converted into numbers that AI can process.

2. **How AI learns** — Training is just "guess the next word, check how wrong you were, adjust, repeat." You'll watch this happen in real time.

3. **Why architecture matters** — You'll train three different AI models (RNN, LSTM, Transformer) and see how each one handles the same task differently.

4. **Why AI sometimes fails** — You'll see models produce repetitive nonsense, and learn the techniques that fix it.

5. **How this connects to ChatGPT** — The exact same principles (next-word prediction, Transformers, training on text) power the AI you use every day — just at a much larger scale.

6. **The limitations and opportunities** — What AI can't do, what biases it inherits from its training data, and where the real opportunities lie.

---

## The Learning Path

Follow the notebooks in order. Each one builds on the previous:

| Notebook | What You'll Learn | Time |
|----------|-------------------|------|
| [01 — Exploring the Data](notebooks/01_exploring_the_data.ipynb) | What training data looks like, why data quality matters | 15 min |
| [02 — How AI Reads Text](notebooks/02_how_ai_reads_text.ipynb) | Tokenization: converting words to numbers | 20 min |
| [03 — Your First AI Model (RNN)](notebooks/03_training_a_simple_rnn.ipynb) | Training a simple model, watching it learn | 30 min |
| [04 — Better Memory (LSTM)](notebooks/04_lstm_better_memory.ipynb) | Why memory matters, comparing LSTM vs RNN | 25 min |
| [05 — The Transformer Revolution](notebooks/05_transformer_revolution.ipynb) | How GPT-2 works, transfer learning, why Transformers changed everything | 30 min |
| [06 — Improving Generation](notebooks/06_improving_generation.ipynb) | Sampling strategies, temperature, fixing repetition | 25 min |
| [07 — Limitations, Ethics & Opportunities](notebooks/07_limitations_ethics_opportunities.ipynb) | Bias, hallucination, environmental cost, real-world applications | 20 min |

**Total learning time: ~3 hours** (you can stop and resume at any point)

---

## Quick Start

### 1. Set up the environment

```bash
conda env create -f environment.yml
conda activate italian-lyrics-ml
```

### 2. Verify everything works

```bash
python scripts/check_environment.py
```

### 3. Prepare the data (if needed)

```bash
python scripts/preprocess_data.py
```

### 4. Start learning

```bash
jupyter lab notebooks/
```

Open `01_exploring_the_data.ipynb` and follow along.

### Or run the full pipeline at once

```bash
python main.py
```

---

## Project Structure

```
marioolm/
├── README.md                  # You are here
├── main.py                    # Full pipeline script (train + generate)
│
├── notebooks/                 # Guided learning path (start here!)
│   ├── 01_exploring_the_data.ipynb
│   ├── 02_how_ai_reads_text.ipynb
│   ├── 03_training_a_simple_rnn.ipynb
│   ├── 04_lstm_better_memory.ipynb
│   ├── 05_transformer_revolution.ipynb
│   ├── 06_improving_generation.ipynb
│   └── 07_limitations_ethics_opportunities.ipynb
│
├── src/                       # Source code (used by notebooks)
│   ├── dataset.py             # Data loading and preparation
│   ├── models.py              # Neural network definitions
│   ├── training.py            # Training loop
│   ├── generation.py          # Text generation functions
│   └── visualization.py       # Charts and visual explanations
│
├── scripts/                   # Utility scripts
│   ├── preprocess_data.py     # Convert raw JSON to training text
│   └── check_environment.py   # Verify your setup
│
├── data/                      # Dataset (not in git — see setup)
│   ├── the_italian_music_dataset.json
│   └── italian_lyrics.txt
│
├── environment.yml            # Conda environment
└── CLAUDE.md                  # AI assistant context
```

---

## The Three Models Explained (No Jargon)

### RNN — The Goldfish
Reads words one at a time and tries to remember what it's read, but its memory fades quickly. Like trying to write a song while only remembering the last few words.

### LSTM — The Note-Taker
Also reads one word at a time, but has a notebook where it writes down important things to remember later. Much better at keeping track of themes across a whole verse.

### Transformer (GPT-2) — The Speed Reader
Reads the entire text at once and understands how every word relates to every other word. This is the architecture behind ChatGPT. It starts with knowledge from millions of English web pages and learns Italian lyrics on top of that.

---

## Scale Comparison: This Project vs. ChatGPT

| | This Project | ChatGPT (GPT-4) |
|---|---|---|
| Training data | 9,135 Italian songs | Trillions of words from the internet |
| Parameters | 124 million (GPT-2) | ~1.8 trillion (estimated) |
| Training time | ~2 hours on laptop | Months on thousands of GPUs |
| Training cost | Free (your electricity) | ~$100 million |
| Languages | Italian only | 100+ languages |

The principles are identical. The scale is what makes the difference.

---

## Requirements

- Python 3.9+
- Conda (recommended) or pip
- ~2 GB free disk space
- No GPU required (runs on CPU, but slower)

See `environment.yml` for full dependency list.

---

## Credits

Dataset: The Italian Music Dataset (~14,679 Italian songs with metadata)

Built as an educational tool to demystify AI for non-technical audiences.
