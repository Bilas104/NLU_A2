"""
=============================================================================
EVALUATE: Name Generation + Quantitative & Qualitative Analysis
=============================================================================
Loads trained models, generates names, and computes:
  - Task 2: Novelty Rate & Diversity Score
  - Task 3: Qualitative analysis with representative samples

Metrics:
  Novelty Rate  = % of generated names NOT in the training set
  Diversity     = # unique generated names / total generated names

Output:
  output/generated_names/     — generated names per model (.txt files)
  output/evaluation_results.txt — full quantitative + qualitative report

Usage:
  python generate_and_evaluate.py
  python generate_and_evaluate.py --num_generate 500 --temperature 0.7
=============================================================================
"""

import torch
import argparse
import os
from collections import Counter

from dataset import load_names, CharVocab
from models import VanillaRNN, BidirectionalLSTM, AttentionRNN


# ============================================================================
# CONFIGURATION
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Generate names and evaluate models")

    parser.add_argument("--data_file", type=str, default="TrainingNames.txt",
                        help="Path to training names file (for novelty check)")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory (must contain models/)")
    parser.add_argument("--num_generate", type=int, default=200,
                        help="Number of names to generate per model")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0.5=conservative, 1.0=diverse)")
    parser.add_argument("--max_len", type=int, default=20,
                        help="Maximum name length during generation")

    # These must match training hyperparameters
    parser.add_argument("--embed_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)

    return parser.parse_args()


# ============================================================================
# LOAD TRAINED MODELS
# ============================================================================

def load_trained_model(model_class, model_name, vocab, args, device):
    """
    Loads a trained model from its checkpoint file.
    Creates the model with the same architecture used during training,
    then loads the saved weights.
    """
    model = model_class(
        vocab_size=vocab.size,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )

    checkpoint_path = os.path.join(args.output_dir, "models", f"{model_name}.pt")

    if not os.path.exists(checkpoint_path):
        print(f"    [ERROR] Checkpoint not found: {checkpoint_path}")
        print(f"    Run train.py first!")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"    Loaded {model_name} (epoch {checkpoint['epoch']}, "
          f"loss {checkpoint['loss']:.4f})")
    return model


# ============================================================================
# QUANTITATIVE EVALUATION (Task 2)
# ============================================================================

def compute_novelty(generated_names, training_names):
    """
    Novelty Rate = percentage of generated names NOT appearing in training set.

    A high novelty rate means the model is generating creative, new names
    rather than simply memorizing the training data.

    Formula: novelty = (1 - |generated ∩ training| / |generated|) × 100
    """
    training_set = set(name.lower() for name in training_names)
    novel_count = sum(1 for name in generated_names
                      if name.lower() not in training_set)
    novelty_rate = (novel_count / len(generated_names)) * 100 if generated_names else 0
    return novelty_rate, novel_count


def compute_diversity(generated_names):
    """
    Diversity = number of unique generated names / total generated names.

    A high diversity score means the model produces varied output rather
    than repeating the same names. Score of 1.0 = all names are unique.

    Formula: diversity = |unique(generated)| / |generated|
    """
    unique_names = set(generated_names)
    diversity = len(unique_names) / len(generated_names) if generated_names else 0
    return diversity, len(unique_names)


def compute_length_stats(generated_names):
    """
    Computes length distribution statistics for generated names.
    Useful for understanding if the model generates realistic-length names.
    """
    if not generated_names:
        return 0, 0, 0

    lengths = [len(name) for name in generated_names]
    return min(lengths), max(lengths), sum(lengths) / len(lengths)


# ============================================================================
# QUALITATIVE ANALYSIS (Task 3)
# ============================================================================

def analyze_quality(generated_names, training_names):
    """
    Performs qualitative analysis on generated names:
      1. Categorizes names by realism (looks like a real name vs not)
      2. Identifies common failure modes
      3. Selects representative samples

    Returns a structured analysis dictionary.
    """
    analysis = {
        "realistic": [],       # names that look plausibly real
        "questionable": [],    # names that are borderline
        "failures": [],        # obvious non-names
        "memorized": [],       # exact copies from training set
        "failure_modes": Counter(),
    }

    training_lower = set(name.lower() for name in training_names)

    for name in generated_names:
        # Check if memorized (exact match from training)
        if name.lower() in training_lower:
            analysis["memorized"].append(name)
            continue

        # Heuristic quality checks
        is_failure = False

        # Check 1: Too short (1-2 chars) or too long (>15 chars)
        if len(name) < 2:
            analysis["failure_modes"]["too_short"] += 1
            is_failure = True
        elif len(name) > 15:
            analysis["failure_modes"]["too_long"] += 1
            is_failure = True

        # Check 2: Excessive repeated characters (e.g., "Aaaaaaa")
        if any(name.count(c) > len(name) * 0.5 for c in set(name)):
            analysis["failure_modes"]["char_repetition"] += 1
            is_failure = True

        # Check 3: No vowels (unpronounceable)
        vowels = set("aeiouAEIOU")
        if not any(c in vowels for c in name):
            analysis["failure_modes"]["no_vowels"] += 1
            is_failure = True

        # Check 4: Consecutive consonant cluster > 4 (unpronounceable)
        consonant_run = 0
        max_consonant_run = 0
        for c in name.lower():
            if c not in "aeiou":
                consonant_run += 1
                max_consonant_run = max(max_consonant_run, consonant_run)
            else:
                consonant_run = 0
        if max_consonant_run > 4:
            analysis["failure_modes"]["consonant_cluster"] += 1
            is_failure = True

        # Check 5: Starts with lowercase (names should be capitalized)
        if name[0].islower():
            analysis["failure_modes"]["wrong_capitalization"] += 1
            is_failure = True

        if is_failure:
            analysis["failures"].append(name)
        elif max_consonant_run > 3 or len(name) < 3:
            analysis["questionable"].append(name)
        else:
            analysis["realistic"].append(name)

    return analysis


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(all_results, training_names, args, report_path):
    """
    Generates a comprehensive evaluation report combining
    quantitative metrics and qualitative analysis.
    """
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("EVALUATION REPORT — CHARACTER-LEVEL NAME GENERATION\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Generation settings:\n")
        f.write(f"  Names generated per model: {args.num_generate}\n")
        f.write(f"  Temperature:               {args.temperature}\n")
        f.write(f"  Max length:                {args.max_len}\n")
        f.write(f"  Training set size:         {len(training_names)}\n\n")

        # ---- QUANTITATIVE SUMMARY TABLE ----
        f.write("─" * 70 + "\n")
        f.write("QUANTITATIVE EVALUATION (Task 2)\n")
        f.write("─" * 70 + "\n\n")

        f.write(f"  {'Model':<22} {'Generated':>10} {'Unique':>8} "
                f"{'Novelty':>9} {'Diversity':>10} {'Avg Len':>8}\n")
        f.write(f"  {'─'*22} {'─'*10} {'─'*8} {'─'*9} {'─'*10} {'─'*8}\n")

        for model_name, results in all_results.items():
            f.write(f"  {model_name:<22} {results['total']:>10} "
                    f"{results['unique']:>8} "
                    f"{results['novelty']:.1f}%{'':<4} "
                    f"{results['diversity']:.4f}{'':<4} "
                    f"{results['avg_len']:.1f}\n")

        # ---- PER-MODEL QUALITATIVE ANALYSIS ----
        for model_name, results in all_results.items():
            f.write(f"\n\n{'═' * 70}\n")
            f.write(f"QUALITATIVE ANALYSIS: {model_name} (Task 3)\n")
            f.write(f"{'═' * 70}\n")

            analysis = results["analysis"]

            # Realism breakdown
            f.write(f"\n  Realism breakdown:\n")
            f.write(f"    Realistic names:     {len(analysis['realistic'])}\n")
            f.write(f"    Questionable names:  {len(analysis['questionable'])}\n")
            f.write(f"    Failed names:        {len(analysis['failures'])}\n")
            f.write(f"    Memorized (exact):   {len(analysis['memorized'])}\n")

            # Failure modes
            if analysis["failure_modes"]:
                f.write(f"\n  Common failure modes:\n")
                for mode, count in analysis["failure_modes"].most_common():
                    f.write(f"    - {mode}: {count} occurrences\n")

            # Representative samples — REALISTIC
            f.write(f"\n  Representative REALISTIC samples (top 20):\n")
            for i, name in enumerate(analysis["realistic"][:20], 1):
                f.write(f"    {i:3d}. {name}\n")

            # Representative samples — FAILURES
            if analysis["failures"]:
                f.write(f"\n  Representative FAILURE samples (up to 10):\n")
                for i, name in enumerate(analysis["failures"][:10], 1):
                    f.write(f"    {i:3d}. {name}\n")

            # Memorized samples
            if analysis["memorized"]:
                f.write(f"\n  Memorized names (exact matches from training, up to 10):\n")
                for i, name in enumerate(analysis["memorized"][:10], 1):
                    f.write(f"    {i:3d}. {name}\n")

        # ---- DISCUSSION TEMPLATE ----
        f.write(f"\n\n{'═' * 70}\n")
        f.write(f"DISCUSSION NOTES (for your report)\n")
        f.write(f"{'═' * 70}\n\n")
        f.write("""
1. REALISM OF GENERATED NAMES:
   - Which model produces the most realistic-sounding names?
   - Do the names follow Indian phonetic patterns?
   - Are generated names pronounceable?

2. NOVELTY vs MEMORIZATION:
   - High novelty means the model is creative, but too high may mean
     it's generating nonsense. The sweet spot is ~70-90%.
   - Low memorization rate suggests the model has learned the underlying
     pattern rather than rote memorization.

3. COMMON FAILURE MODES:
   - Vanilla RNN often suffers from character repetition (e.g., "Aaaaarav")
     because it has no gating mechanism to control information flow.
   - BLSTM may produce less coherent names during generation because
     it was trained with bidirectional context but generates with only
     forward context (train-generate mismatch).
   - Attention RNN should handle longer names better because it can
     look back at earlier characters to maintain consistency.

4. MODEL COMPARISON:
   - Vanilla RNN: fastest training, fewest parameters, but lower quality
   - BLSTM: most parameters, train-generate mismatch is a key weakness
   - Attention RNN: best at maintaining coherence over longer names,
     slightly slower due to attention computation

5. TEMPERATURE EFFECT:
   - Low temperature (0.3-0.5): conservative, repetitive, more memorization
   - Medium temperature (0.7-0.9): balanced creativity and coherence
   - High temperature (1.0+): very diverse but more failures
""")

    print(f"\n  Report saved to {report_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()

    # Create output dirs
    gen_dir = os.path.join(args.output_dir, "generated_names")
    os.makedirs(gen_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print(f"NAME GENERATION & EVALUATION")
    print(f"{'=' * 60}")
    print(f"  Device: {device}")
    print(f"  Generating {args.num_generate} names per model")
    print(f"  Temperature: {args.temperature}")

    # ---- Load training data (for novelty comparison) ----
    print(f"\n[1/4] Loading training data...")
    training_names = load_names(args.data_file)
    vocab = CharVocab(training_names)

    # ---- Load all models ----
    print(f"\n[2/4] Loading trained models...")
    model_classes = {
        "VanillaRNN": VanillaRNN,
        "BidirectionalLSTM": BidirectionalLSTM,
        "AttentionRNN": AttentionRNN,
    }

    models = {}
    for name, cls in model_classes.items():
        model = load_trained_model(cls, name, vocab, args, device)
        if model is not None:
            models[name] = model

    if not models:
        print("[ERROR] No models loaded. Run train.py first!")
        return

    # ---- Generate names and evaluate ----
    print(f"\n[3/4] Generating and evaluating...")
    all_results = {}

    for model_name, model in models.items():
        print(f"\n  --- {model_name} ---")

        # Generate names
        generated = model.generate(
            vocab,
            max_len=args.max_len,
            num_names=args.num_generate,
            temperature=args.temperature,
            device=device
        )
        print(f"    Generated: {len(generated)} names")

        # Save generated names to file
        gen_path = os.path.join(gen_dir, f"{model_name}_generated.txt")
        with open(gen_path, "w") as f:
            for name in generated:
                f.write(name + "\n")
        print(f"    Saved to: {gen_path}")

        # Quantitative metrics
        novelty_rate, novel_count = compute_novelty(generated, training_names)
        diversity, unique_count = compute_diversity(generated)
        min_len, max_len, avg_len = compute_length_stats(generated)

        print(f"    Novelty:   {novelty_rate:.1f}% ({novel_count}/{len(generated)} novel)")
        print(f"    Diversity: {diversity:.4f} ({unique_count}/{len(generated)} unique)")
        print(f"    Lengths:   min={min_len}, max={max_len}, avg={avg_len:.1f}")

        # Qualitative analysis
        analysis = analyze_quality(generated, training_names)
        print(f"    Realistic: {len(analysis['realistic'])}, "
              f"Failures: {len(analysis['failures'])}, "
              f"Memorized: {len(analysis['memorized'])}")

        # Print a few samples
        print(f"    Sample names: {generated[:10]}")

        all_results[model_name] = {
            "names": generated,
            "total": len(generated),
            "unique": unique_count,
            "novelty": novelty_rate,
            "diversity": diversity,
            "min_len": min_len,
            "max_len": max_len,
            "avg_len": avg_len,
            "analysis": analysis,
        }

    # ---- Generate full report ----
    print(f"\n[4/4] Generating evaluation report...")
    report_path = os.path.join(args.output_dir, "evaluation_results.txt")
    generate_report(all_results, training_names, args, report_path)

    # ---- Print summary table ----
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n  {'Model':<22} {'Novelty':>9} {'Diversity':>10} {'Realistic':>10}")
    print(f"  {'─'*22} {'─'*9} {'─'*10} {'─'*10}")
    for model_name, r in all_results.items():
        realistic_pct = len(r['analysis']['realistic']) / r['total'] * 100
        print(f"  {model_name:<22} {r['novelty']:>7.1f}% {r['diversity']:>10.4f} "
              f"{realistic_pct:>8.1f}%")

    print(f"\n  All outputs saved in {args.output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
