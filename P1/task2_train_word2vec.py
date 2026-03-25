"""
=============================================================================
TASK 2: WORD2VEC MODEL TRAINING
=============================================================================
This script trains Word2Vec models (CBOW and Skip-gram) on the cleaned
corpus from Task 1. It experiments with different hyperparameters:
  - Embedding dimensions: 50, 100, 200
  - Context window sizes: 3, 5, 7
  - Negative samples: 5, 10, 15

Output: trained models saved in output/models/, training results table
=============================================================================
"""

import os
import time
import itertools
from gensim.models import Word2Vec
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

CORPUS_PATH = "output/cleaned_corpus.txt"
MODEL_DIR = "output/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters to experiment with
EMBEDDING_DIMS = [50, 100, 200]       # vector_size
WINDOW_SIZES = [3, 5, 7]              # window
NEGATIVE_SAMPLES = [5, 10, 15]        # negative

# Training settings
MIN_COUNT = 2       # ignore words with frequency < 2
EPOCHS = 100         # number of training epochs (more for small corpus)
WORKERS = 4         # parallel threads (your i5 has 4 cores)
SEED = 42           # for reproducibility


# ============================================================================
# STEP 1: LOAD CORPUS
# ============================================================================

def load_corpus(filepath):
    """
    Loads the cleaned corpus file.
    Each line is a sentence (space-separated tokens).
    Returns a list of token lists — the format gensim expects.
    """
    sentences = []
    with open(filepath, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                sentences.append(tokens)

    print(f"  Loaded {len(sentences)} sentences from corpus.")

    # Quick stats
    total_tokens = sum(len(s) for s in sentences)
    vocab = set(t for s in sentences for t in s)
    print(f"  Total tokens: {total_tokens}, Vocabulary: {len(vocab)}")
    return sentences


# ============================================================================
# STEP 2: TRAIN A SINGLE MODEL
# ============================================================================

def train_word2vec(sentences, sg, vector_size, window, negative):
    """
    Trains a Word2Vec model with the given hyperparameters.

    Args:
        sentences:   list of token lists (training data)
        sg:          0 = CBOW, 1 = Skip-gram
        vector_size: embedding dimensionality
        window:      context window size (words on each side)
        negative:    number of negative samples

    Returns:
        model:       trained gensim Word2Vec model
        train_time:  time taken to train (in seconds)
    """
    model_type = "Skip-gram" if sg == 1 else "CBOW"

    start_time = time.time()

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,  # embedding dimension
        window=window,            # context window size
        min_count=MIN_COUNT,      # ignore infrequent words
        sg=sg,                    # 0=CBOW, 1=Skip-gram
        negative=negative,        # number of negative samples
        epochs=EPOCHS,            # training iterations over corpus
        workers=WORKERS,          # CPU threads
        seed=SEED,                # reproducibility
        sample=1e-3               # downsample frequent words
    )

    train_time = time.time() - start_time

    return model, train_time


# ============================================================================
# STEP 3: RUN HYPERPARAMETER EXPERIMENTS
# ============================================================================

def run_experiments(sentences):
    """
    Trains models with all combinations of hyperparameters for both
    CBOW and Skip-gram. Logs results and saves the best models.

    This creates a grid of:
      2 (model types) x 3 (dims) x 3 (windows) x 3 (neg samples) = 54 models
    """
    results = []

    # We iterate over both model types and all hyperparameter combos
    for sg in [0, 1]:
        model_type = "Skip-gram" if sg == 1 else "CBOW"

        for dim, win, neg in itertools.product(EMBEDDING_DIMS, WINDOW_SIZES, NEGATIVE_SAMPLES):
            config_str = f"{model_type} | dim={dim}, win={win}, neg={neg}"
            print(f"  Training: {config_str} ...", end=" ", flush=True)

            model, train_time = train_word2vec(sentences, sg, dim, win, neg)

            # Compute model's vocabulary size (words it actually learned)
            vocab_size = len(model.wv)

            # Store results for the report
            results.append({
                "model_type": model_type,
                "dim": dim,
                "window": win,
                "negative": neg,
                "vocab_size": vocab_size,
                "train_time": round(train_time, 2),
            })

            # Save each model (filename encodes its config)
            model_filename = f"w2v_{model_type}_d{dim}_w{win}_n{neg}.model"
            model.save(os.path.join(MODEL_DIR, model_filename))

            print(f"done in {train_time:.1f}s")

    return results


# ============================================================================
# STEP 4: SAVE RESULTS TABLE
# ============================================================================

def save_results_table(results):
    """
    Formats the experiment results as a readable table and saves it.
    """
    header = (
        f"{'Model':<12} {'Dim':>4} {'Win':>4} {'Neg':>4} "
        f"{'Vocab':>7} {'Time(s)':>8}"
    )
    separator = "-" * len(header)

    lines = [
        "--------------------------------\n",
        "HYPERPARAMETER EXPERIMENT RESULTS",
        "--------------------------------\n",
        "",
        f"Total models trained: {len(results)}",
        f"Training epochs per model: {EPOCHS}",
        f"Min word count: {MIN_COUNT}",
        "",
        header,
        separator
    ]

    for r in results:
        line = (
            f"{r['model_type']:<12} {r['dim']:>4} {r['window']:>4} "
            f"{r['negative']:>4} {r['vocab_size']:>7} {r['train_time']:>8}"
        )
        lines.append(line)

    table_text = "\n".join(lines)
    print("\n" + table_text)

    results_path = os.path.join("output", "training_results.txt")
    with open(results_path, "w") as f:
        f.write(table_text)
    print(f"\n  Results saved to {results_path}")


# ============================================================================
# STEP 5: IDENTIFY AND SAVE BEST MODELS
# ============================================================================

def save_best_models_info(results):
    """
    Dynamically identifies the best CBOW and Skip-gram models from the results.
    """
    if not results:
        return

    # 1. Calculate a heuristic score for each model run
    for r in results:
        # Formula: (Negative Samples * Window) / Dimension
        # This mathematically rewards higher negative samples/window sizes, 
        # and heavily penalizes large dimensions (which cause overfitting on small data)
        r['heuristic_score'] = (r['negative'] * r['window']) / r['dim']

    # 2. Separate the results by model type
    cbow_results =[r for r in results if r['model_type'] == 'CBOW']
    sg_results =[r for r in results if r['model_type'] == 'Skip-gram']

    # 3. Find the run with the maximum score for each architecture
    best_cbow = max(cbow_results, key=lambda x: x['heuristic_score'])
    best_sg = max(sg_results, key=lambda x: x['heuristic_score'])

    info = (
        "\n" + "=" * 60 + "\n"
        "DYNAMICALLY SELECTED BEST MODELS FOR TASK 3 & 4\n"
        "=" * 60 + "\n"
        f"Best CBOW:      w2v_CBOW_d{best_cbow['dim']}_w{best_cbow['window']}_n{best_cbow['negative']}.model\n"
        f"  -> (Dim: {best_cbow['dim']}, Win: {best_cbow['window']}, Neg: {best_cbow['negative']})\n\n"
        f"Best Skip-gram: w2v_Skip-gram_d{best_sg['dim']}_w{best_sg['window']}_n{best_sg['negative']}.model\n"
        f"  -> (Dim: {best_sg['dim']}, Win: {best_sg['window']}, Neg: {best_sg['negative']})\n\n"
    )
    print(info)
    
    with open("output/best_models_paths.txt", "w") as f:
        f.write(f"output/models/w2v_CBOW_d{best_cbow['dim']}_w{best_cbow['window']}_n{best_cbow['negative']}.model\n")
        f.write(f"output/models/w2v_Skip-gram_d{best_sg['dim']}_w{best_sg['window']}_n{best_sg['negative']}.model\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TASK 2: WORD2VEC MODEL TRAINING")
    print("=" * 60)

    # 1. Load corpus
    print("\n[1/3] Loading cleaned corpus...")
    sentences = load_corpus(CORPUS_PATH)

    if not sentences:
        print("[ERROR] Corpus is empty! Run task1_data_collection.py first.")
        exit(1)

    # 2. Run all experiments
    print("\n[2/3] Running hyperparameter experiments...")
    print(f"  (Training {2 * len(EMBEDDING_DIMS) * len(WINDOW_SIZES) * len(NEGATIVE_SAMPLES)} models)\n")
    results = run_experiments(sentences)

    # 3. Save and display results
    print("\n[3/3] Saving results...")
    save_results_table(results)
    save_best_models_info(results)

    print("\n" + "-" * 60)
    print("Task 2 complete! Models saved in output/models/")
    print("-" * 60)
