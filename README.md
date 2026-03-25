# CSL 7640: Natural Language Understanding — Assignment 2

## Problem 1: Learning Word Embeddings from IIT Jodhpur Data

Trains Word2Vec (CBOW & Skip-gram) on text scraped from IIT Jodhpur sources and analyzes the learned embeddings.

### Setup

```bash
pip install requests beautifulsoup4 gensim nltk matplotlib wordcloud scikit-learn numpy PyPDF2
```

### How to Run

Run the scripts in order — each one depends on the output of the previous:

```bash
# Step 1: Scrape data, preprocess, generate word cloud and corpus stats
python task1_data_collection_v2.py

# Step 2: Train CBOW + Skip-gram models (54 hyperparameter combinations)
python task2_train_word2vec.py

# Step 3: Nearest neighbors + analogy experiments
python task3_semantic_analysis.py

# Step 4: PCA and t-SNE visualizations
python task4_visualization.py
```

### Output

All outputs are saved in the `output/` directory:

- `cleaned_corpus.txt` — preprocessed corpus (one sentence per line)
- `corpus_stats.txt` — token counts, vocabulary size, top-30 words
- `wordcloud.png` — word cloud of the corpus
- `source_log.txt` — list of all crawled URLs
- `models/` — 54 trained Word2Vec models
- `training_results.txt` — hyperparameter experiment table
- `semantic_analysis_results.txt` — nearest neighbors and analogy results
- `visualizations/` — PCA and t-SNE plots

### Notes

- The crawler stays within `*.iitj.ac.in` and follows links up to 2 levels deep. Crawling takes roughly 10–20 minutes.
- The academic regulations PDF is auto-downloaded from the URL in the seed list. If the URL changes, update `SEED_URLS` in `task1_data_collection_v2.py`.
- If some cluster words in Task 4 are missing from your vocabulary, edit `WORD_CLUSTERS` in `task4_visualization.py` to match words that actually appear in your corpus.

---

## Problem 2: Character-Level Name Generation Using RNN Variants

Implements three recurrent architectures from scratch (no `nn.RNN`/`nn.LSTM`) for generating Indian names and compares them quantitatively and qualitatively.

### Setup

This is intended to run on a GPU server. On the server:

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Load CUDA (if using a module system like on HPC clusters)
module load cuda/12.1.1

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib
```

For CPU-only (runs slower but works anywhere):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib
```

### How to Run

```bash
# Step 1: Train all three models (Vanilla RNN, BiLSTM, Attention RNN)
python train.py

# Step 2: Generate names and compute evaluation metrics
python generate_and_evaluate.py
```

Optional flags for `train.py`:

```bash
python train.py --epochs 200 --hidden_size 256 --lr 0.001
```

Optional flags for `generate_and_evaluate.py`:

```bash
python generate_and_evaluate.py --num_generate 500 --temperature 0.7
```

### Output

All outputs are saved in the `output/` directory:

- `models/` — saved checkpoints for all three models (`.pt` files)
- `loss_curves.png` — training loss comparison across models
- `training_log.txt` — hyperparameters and training summary
- `generated_names/` — generated name samples per model (`.txt` files)
- `evaluation_results.txt` — novelty, diversity, and qualitative analysis

### File Structure

| File | Purpose |
|---|---|
| `TrainingNames.txt` | 1000 Indian names used for training |
| `dataset.py` | Character vocabulary, encoding/decoding, PyTorch Dataset |
| `models.py` | Vanilla RNN, Bidirectional LSTM, Attention RNN (from scratch) |
| `train.py` | Training loop with checkpointing and loss curves |
| `generate_and_evaluate.py` | Name generation + novelty/diversity metrics |

### Hardware Used

- **Problem 1**: Any laptop (CPU-only, gensim handles it)
- **Problem 2**: NVIDIA A30 (24 GB VRAM), 32 GB RAM — training takes ~2 minutes total for all three models
