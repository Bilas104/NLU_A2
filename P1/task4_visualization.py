"""
=============================================================================
TASK 4: EMBEDDING VISUALIZATION
=============================================================================
This script visualizes Word2Vec embeddings using PCA and t-SNE.
It projects high-dimensional word vectors into 2D space and plots
clusters of semantically related words.

Clusters visualized:
  - Academic terms (research, thesis, publication, etc.)
  - Student life (student, hostel, campus, etc.)
  - Departments (cse, ee, me, math, etc.)
  - Degree programs (btech, mtech, phd, msc, etc.)

Output: Multiple PNG plots in output/visualizations/
=============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

# ============================================================================
# CONFIGURATION
# ============================================================================

CBOW_MODEL_PATH = "output/models/w2v_CBOW_d100_w5_n5.model"
SKIPGRAM_MODEL_PATH = "output/models/w2v_Skip-gram_d100_w5_n5.model"

VIZ_DIR = "output/visualizations"
os.makedirs(VIZ_DIR, exist_ok=True)

# Define semantic clusters to visualize
# Each cluster is a group of words we expect to be close in embedding space
# NOTE: Some words may not be in your vocabulary — the script handles this
WORD_CLUSTERS = {
    "Academic / Research": [
        "research", "thesis", "publication", "paper", "conference",
        "journal", "project", "innovation", "laboratory", "scholar"
    ],
    "Student Life": [
        "student", "hostel", "campus", "library", "semester",
        "examination", "attendance", "placement", "club", "fest"
    ],
    "Departments": [
        "cse", "computer", "electrical", "mechanical", "mathematics",
        "physics", "chemistry", "biology", "civil", "humanities"
    ],
    "Degree Programs": [
        "btech", "mtech", "phd", "msc", "diploma",
        "undergraduate", "postgraduate", "doctoral", "degree", "programme"
    ],
}

# Colors for each cluster (matplotlib color names)
CLUSTER_COLORS = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]


# ============================================================================
# STEP 1: EXTRACT WORD VECTORS FOR SELECTED WORDS
# ============================================================================

def get_word_vectors(model, word_clusters):
    """
    Extracts embedding vectors for words in each cluster.
    Filters out words not found in the model's vocabulary.

    Returns:
      - vectors: numpy array of shape (n_words, embedding_dim)
      - labels: list of word strings
      - cluster_ids: list of cluster indices (for coloring)
      - cluster_names: list of cluster name strings
    """
    vectors = []
    labels = []
    cluster_ids = []
    cluster_names = list(word_clusters.keys())

    for cluster_idx, (cluster_name, words) in enumerate(word_clusters.items()):
        found_count = 0
        for word in words:
            if word in model.wv:
                vectors.append(model.wv[word])
                labels.append(word)
                cluster_ids.append(cluster_idx)
                found_count += 1

        print(f"    {cluster_name}: {found_count}/{len(words)} words found")

    vectors = np.array(vectors)
    print(f"    Total words for visualization: {len(labels)}")
    return vectors, labels, cluster_ids, cluster_names


# ============================================================================
# STEP 2: PCA VISUALIZATION
# ============================================================================

def plot_pca(vectors, labels, cluster_ids, cluster_names, model_name, save_path):
    """
    Reduces embeddings to 2D using PCA (Principal Component Analysis).

    PCA finds the directions of maximum variance in the data.
    It's deterministic and fast, but may not preserve local structure
    as well as t-SNE.
    """
    # Fit PCA to reduce from embedding_dim → 2 dimensions
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(vectors)

    # Report explained variance (how much info is preserved in 2D)
    var_explained = pca.explained_variance_ratio_
    print(f"    PCA variance explained: PC1={var_explained[0]:.3f}, PC2={var_explained[1]:.3f}")
    print(f"    Total variance retained: {sum(var_explained):.3f} ({sum(var_explained)*100:.1f}%)")

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(14, 10))

    for cluster_idx, cluster_name in enumerate(cluster_names):
        # Get indices of words belonging to this cluster
        mask = [i for i, cid in enumerate(cluster_ids) if cid == cluster_idx]
        if not mask:
            continue

        x = coords_2d[mask, 0]
        y = coords_2d[mask, 1]

        ax.scatter(x, y,
                   c=CLUSTER_COLORS[cluster_idx],
                   label=cluster_name,
                   s=100, alpha=0.7, edgecolors='white', linewidths=0.5)

        # Add word labels next to each point
        for i, idx in enumerate(mask):
            ax.annotate(labels[idx],
                        (coords_2d[idx, 0], coords_2d[idx, 1]),
                        fontsize=9, fontweight='bold',
                        xytext=(5, 5), textcoords='offset points',
                        color=CLUSTER_COLORS[cluster_idx])

    ax.set_title(f"PCA Projection — {model_name}\n"
                 f"(Variance explained: {sum(var_explained)*100:.1f}%)",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% variance)")
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


# ============================================================================
# STEP 3: t-SNE VISUALIZATION
# ============================================================================

def plot_tsne(vectors, labels, cluster_ids, cluster_names, model_name, save_path):
    """
    Reduces embeddings to 2D using t-SNE (t-distributed Stochastic
    Neighbor Embedding).

    t-SNE preserves local neighborhood structure better than PCA.
    It's non-deterministic and slower, but often produces more
    visually interpretable clusters.

    Key parameter: perplexity ≈ expected number of neighbors.
    Should be less than the number of data points.
    """
    # Adjust perplexity based on number of points
    # Rule of thumb: perplexity should be < n_samples / 3
    n_samples = len(vectors)
    perplexity = min(15, max(5, n_samples // 4))

    print(f"    t-SNE with perplexity={perplexity}, n_samples={n_samples}")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        max_iter=1000,          # iterations for optimization
        learning_rate='auto'  # let sklearn choose
    )
    coords_2d = tsne.fit_transform(vectors)

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(14, 10))

    for cluster_idx, cluster_name in enumerate(cluster_names):
        mask = [i for i, cid in enumerate(cluster_ids) if cid == cluster_idx]
        if not mask:
            continue

        x = coords_2d[mask, 0]
        y = coords_2d[mask, 1]

        ax.scatter(x, y,
                   c=CLUSTER_COLORS[cluster_idx],
                   label=cluster_name,
                   s=100, alpha=0.7, edgecolors='white', linewidths=0.5)

        for i, idx in enumerate(mask):
            ax.annotate(labels[idx],
                        (coords_2d[idx, 0], coords_2d[idx, 1]),
                        fontsize=9, fontweight='bold',
                        xytext=(5, 5), textcoords='offset points',
                        color=CLUSTER_COLORS[cluster_idx])

    ax.set_title(f"t-SNE Projection — {model_name}\n"
                 f"(perplexity={perplexity})",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


# ============================================================================
# STEP 4: COMPARISON PLOT (CBOW vs Skip-gram side by side)
# ============================================================================

def plot_comparison(cbow_model, sg_model, word_clusters, save_path):
    """
    Creates a side-by-side PCA comparison of CBOW vs Skip-gram.
    This is useful for the report to visually show differences.
    """
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    for ax, (model_name, model) in zip(axes, [("CBOW", cbow_model), ("Skip-gram", sg_model)]):
        vectors, labels, cluster_ids, cluster_names = get_word_vectors(model, word_clusters)

        if len(vectors) < 2:
            ax.text(0.5, 0.5, "Not enough words in vocabulary",
                    ha='center', va='center', fontsize=12)
            continue

        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(vectors)

        for cluster_idx, cluster_name in enumerate(cluster_names):
            mask = [i for i, cid in enumerate(cluster_ids) if cid == cluster_idx]
            if not mask:
                continue
            x = coords_2d[mask, 0]
            y = coords_2d[mask, 1]
            ax.scatter(x, y, c=CLUSTER_COLORS[cluster_idx],
                       label=cluster_name, s=80, alpha=0.7,
                       edgecolors='white', linewidths=0.5)
            for i, idx in enumerate(mask):
                ax.annotate(labels[idx],
                            (coords_2d[idx, 0], coords_2d[idx, 1]),
                            fontsize=8, xytext=(4, 4),
                            textcoords='offset points',
                            color=CLUSTER_COLORS[cluster_idx])

        var_exp = sum(pca.explained_variance_ratio_) * 100
        ax.set_title(f"{model_name}\n(PCA variance: {var_exp:.1f}%)",
                     fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("CBOW vs Skip-gram — Embedding Comparison (PCA)",
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TASK 4: EMBEDDING VISUALIZATION")
    print("=" * 60)

    # 1. Load models
    print("\n[1/4] Loading models...")
    cbow_model = Word2Vec.load(CBOW_MODEL_PATH)
    sg_model = Word2Vec.load(SKIPGRAM_MODEL_PATH)

    # 2. PCA plots
    for model_name, model in [("CBOW", cbow_model), ("Skip-gram", sg_model)]:
        print(f"\n[2/4] PCA visualization for {model_name}...")
        vectors, labels, cluster_ids, cluster_names = get_word_vectors(model, WORD_CLUSTERS)

        if len(vectors) >= 2:
            save_path = os.path.join(VIZ_DIR, f"pca_{model_name.lower()}.png")
            plot_pca(vectors, labels, cluster_ids, cluster_names, model_name, save_path)
        else:
            print(f"    [SKIP] Not enough words found for {model_name}")

    # 3. t-SNE plots
    for model_name, model in [("CBOW", cbow_model), ("Skip-gram", sg_model)]:
        print(f"\n[3/4] t-SNE visualization for {model_name}...")
        vectors, labels, cluster_ids, cluster_names = get_word_vectors(model, WORD_CLUSTERS)

        if len(vectors) >= 2:
            save_path = os.path.join(VIZ_DIR, f"tsne_{model_name.lower()}.png")
            plot_tsne(vectors, labels, cluster_ids, cluster_names, model_name, save_path)
        else:
            print(f"    [SKIP] Not enough words found for {model_name}")

    # 4. Side-by-side comparison
    print(f"\n[4/4] Generating comparison plot...")
    comparison_path = os.path.join(VIZ_DIR, "cbow_vs_skipgram_comparison.png")
    plot_comparison(cbow_model, sg_model, WORD_CLUSTERS, comparison_path)

    print("\n" + "=" * 60)
    print("Task 4 complete! Check output/visualizations/")
    print("=" * 60)
