"""
=============================================================================
TASK 3: SEMANTIC ANALYSIS
=============================================================================
This script performs semantic analysis on trained Word2Vec models:
  1. Finds top-5 nearest neighbors for specified query words
  2. Runs analogy experiments (e.g., UG:BTech :: PG:?)
  3. Compares CBOW vs Skip-gram results

Uses cosine similarity (built into gensim's most_similar method).
Output: semantic_analysis_results.txt
=============================================================================
"""

import os
from gensim.models import Word2Vec

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths to best CBOW and Skip-gram models from Task 2
# UPDATE these if you chose different hyperparameters
CBOW_MODEL_PATH = "output/models/w2v_CBOW_d100_w5_n5.model"
SKIPGRAM_MODEL_PATH = "output/models/w2v_Skip-gram_d100_w5_n5.model"

# Query words for nearest neighbor search (as specified in assignment)
QUERY_WORDS = ["research", "student", "phd", "exam"]

ANALOGIES = [
    ("ug", "btech", "pg", "UG : BTech :: PG : ?"),
    ("professor", "teaching", "student", "professor : teaching :: student : ?"),
    ("cse", "computer", "ee", "CSE : computer :: EE : ?"),
    ("director", "college", "hod", "director : college :: HOD : ?"),
]

OUTPUT_PATH = "output/semantic_analysis_results.txt"


# ============================================================================
# STEP 1: LOAD MODELS
# ============================================================================

def load_models():
    """
    Loads the pre-trained CBOW and Skip-gram models from Task 2.
    Returns a dict of {name: model} for easy iteration.
    """
    models = {}

    print("  Loading CBOW model...")
    models["CBOW"] = Word2Vec.load(CBOW_MODEL_PATH)
    print(f"    Vocabulary size: {len(models['CBOW'].wv)}")

    print("  Loading Skip-gram model...")
    models["Skip-gram"] = Word2Vec.load(SKIPGRAM_MODEL_PATH)
    print(f"    Vocabulary size: {len(models['Skip-gram'].wv)}")

    return models


# ============================================================================
# STEP 2: NEAREST NEIGHBORS (Cosine Similarity)
# ============================================================================

def find_nearest_neighbors(models, query_words, topn=5):
    """
    For each query word, finds the top-N most similar words using
    cosine similarity in the embedding space.

    Cosine similarity = dot(u, v) / (||u|| * ||v||)
    Gensim's most_similar() uses this internally.

    Returns formatted results string for the report.
    """
    results_text = ""

    for word in query_words:
        results_text += f"\n{'─' * 50}\n"
        results_text += f"Query word: '{word}'\n"
        results_text += f"{'─' * 50}\n"

        for model_name, model in models.items():
            results_text += f"\n  [{model_name}]\n"

            # Check if word exists in model's vocabulary
            if word not in model.wv:
                results_text += f"    '{word}' not in vocabulary!\n"
                results_text += f"    (Try a different word or collect more data)\n"
                continue

            # Get top-5 nearest neighbors
            neighbors = model.wv.most_similar(word, topn=topn)

            results_text += f"  {'Rank':<6} {'Word':<20} {'Cosine Similarity'}\n"
            results_text += f"  {'─'*6} {'─'*20} {'─'*18}\n"

            for rank, (neighbor, similarity) in enumerate(neighbors, 1):
                results_text += f"  {rank:<6} {neighbor:<20} {similarity:.4f}\n"

    return results_text


# ============================================================================
# STEP 3: ANALOGY EXPERIMENTS
# ============================================================================

def run_analogies(models, analogies):
    """
    Performs word analogy tasks using the vector arithmetic method:
        A is to B as C is to ?
        Answer = vector(B) - vector(A) + vector(C)

    For example: UG:BTech :: PG:?
        ? = vec(BTech) - vec(UG) + vec(PG)

    This is the classic Word2Vec analogy test from the original paper.
    Returns formatted results string.
    """
    results_text = ""

    for word_a, word_b, word_c, description in analogies:
        results_text += f"\n{'─' * 50}\n"
        results_text += f"Analogy: {description}\n"
        results_text += f"  Method: vec({word_b}) - vec({word_a}) + vec({word_c})\n"
        results_text += f"{'─' * 50}\n"

        for model_name, model in models.items():
            results_text += f"\n  [{model_name}]\n"

            # Check all three words exist in vocabulary
            missing = [w for w in [word_a, word_b, word_c] if w not in model.wv]
            if missing:
                results_text += f"    Missing words: {missing}\n"
                results_text += f"    Cannot perform this analogy.\n"
                continue

            # Gensim's most_similar with positive/negative does the vector math:
            #   positive = [word_b, word_c] → adds their vectors
            #   negative = [word_a]         → subtracts its vector
            try:
                predictions = model.wv.most_similar(
                    positive=[word_b, word_c],
                    negative=[word_a],
                    topn=5
                )

                results_text += f"  {'Rank':<6} {'Predicted Word':<20} {'Score'}\n"
                results_text += f"  {'─'*6} {'─'*20} {'─'*10}\n"

                for rank, (word, score) in enumerate(predictions, 1):
                    results_text += f"  {rank:<6} {word:<20} {score:.4f}\n"

            except KeyError as e:
                results_text += f"    Error: {e}\n"

    return results_text

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TASK 3: SEMANTIC ANALYSIS")
    print("=" * 60)

    # 1. Load models
    print("\n[1/3] Loading trained models...")
    models = load_models()

    # 2. Nearest neighbors
    print("\n[2/3] Finding nearest neighbors...")
    nn_results = find_nearest_neighbors(models, QUERY_WORDS)
    print(nn_results)

    # 3. Analogies
    print("\n[3/3] Running analogy experiments...")
    analogy_results = run_analogies(models, ANALOGIES)
    print(analogy_results)

    # 4. Save everything
    full_output = (
        "=" * 60 + "\n"
        "SEMANTIC ANALYSIS RESULTS\n"
        "=" * 60 + "\n"
        "\n\n━━━ PART 1: NEAREST NEIGHBORS ━━━\n"
        + nn_results
        + "\n\n━━━ PART 2: ANALOGY EXPERIMENTS ━━━\n"
        + analogy_results
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(full_output)

    print(f"\n  Full results saved to {OUTPUT_PATH}")
    print("\n" + "=" * 60)
    print("Task 3 complete!")
    print("=" * 60)
