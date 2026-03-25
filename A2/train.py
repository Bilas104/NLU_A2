"""
=============================================================================
TRAIN: Training Script for All Three Models
=============================================================================
Trains VanillaRNN, BidirectionalLSTM, and AttentionRNN on the names dataset.

For each model:
  - Trains with teacher forcing and cross-entropy loss
  - Logs training loss per epoch
  - Saves model checkpoint + loss curve plot
  - Reports training time

Output:
  output/models/          — saved model checkpoints (.pt files)
  output/loss_curves.png  — training loss comparison plot
  output/training_log.txt — full training log with hyperparameters

Usage:
  python train.py
  python train.py --epochs 200 --hidden_size 256
=============================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import time
import os
import matplotlib.pyplot as plt

from dataset import load_names, CharVocab, NamesDataset, collate_fn
from models import VanillaRNN, BidirectionalLSTM, AttentionRNN, print_model_summary


# ============================================================================
# CONFIGURATION (command-line overridable)
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train character-level name generation models")

    # Data
    parser.add_argument("--data_file", type=str, default="TrainingNames.txt",
                        help="Path to training names file")

    # Hyperparameters
    parser.add_argument("--embed_size", type=int, default=64,
                        help="Character embedding dimension")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="RNN hidden state dimension")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of stacked RNN layers")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.003,
                        help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=5.0,
                        help="Gradient clipping max norm (prevents exploding gradients)")

    # Output
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory for saving models and plots")

    return parser.parse_args()


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_model(model, model_name, train_loader, vocab, args, device):
    """
    Trains a single model using teacher forcing.

    Teacher forcing: at each timestep, the model receives the GROUND TRUTH
    previous character as input (not its own prediction). This stabilizes
    training by preventing error accumulation.

    Loss: Cross-entropy between predicted next-character distribution and
    the actual next character. Padding positions are ignored.

    Args:
        model:        nn.Module — the model to train
        model_name:   str — name for logging/saving
        train_loader: DataLoader — training data
        vocab:        CharVocab — character vocabulary
        args:         parsed arguments
        device:       torch device

    Returns:
        loss_history: list of average loss per epoch (for plotting)
    """
    model = model.to(device)

    # Adam optimizer — works well for RNNs, adaptive learning rates
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Cross-entropy loss, ignoring padding tokens (index 0)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    # Learning rate scheduler — reduce LR when loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, verbose=True
    )

    loss_history = []
    best_loss = float('inf')
    model_save_path = os.path.join(args.output_dir, "models", f"{model_name}.pt")

    print(f"\n{'─' * 50}")
    print(f"Training {model_name}")
    print(f"{'─' * 50}")
    print(f"  Device: {device}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"  LR: {args.lr}, Grad clip: {args.grad_clip}")
    print()

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for inputs, targets, lengths in train_loader:
            inputs = inputs.to(device)    # (batch, seq_len)
            targets = targets.to(device)  # (batch, seq_len)
            lengths = lengths.to(device)

            # Forward pass — model predicts next character at each position
            logits = model(inputs, lengths)  # (batch, seq_len, vocab_size)

            # Reshape for cross-entropy: (batch*seq_len, vocab_size) vs (batch*seq_len,)
            logits_flat = logits.reshape(-1, vocab.size)
            targets_flat = targets.reshape(-1)

            # Compute loss (padding positions automatically ignored)
            loss = criterion(logits_flat, targets_flat)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients to prevent exploding gradient problem
            # This is especially important for vanilla RNNs
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        # Average loss for this epoch
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)

        # Step the LR scheduler
        scheduler.step(avg_loss)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
                'vocab_size': vocab.size,
                'embed_size': args.embed_size,
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers,
            }, model_save_path)

        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:4d}/{args.epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Best: {best_loss:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {elapsed:.0f}s")

    total_time = time.time() - start_time
    print(f"\n  Training complete: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Model saved: {model_save_path}")

    return loss_history, total_time


# ============================================================================
# PLOTTING
# ============================================================================

def plot_loss_curves(all_losses, model_names, save_path):
    """
    Plots training loss curves for all models on one figure.
    Useful for visual comparison in the report.
    """
    plt.figure(figsize=(10, 6))

    colors = ['#e74c3c', '#2ecc71', '#3498db']
    for losses, name, color in zip(all_losses, model_names, colors):
        plt.plot(range(1, len(losses) + 1), losses, label=name,
                 linewidth=2, color=color, alpha=0.85)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.title("Training Loss Comparison — All Models", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Loss curves saved to {save_path}")


# ============================================================================
# TRAINING LOG
# ============================================================================

def save_training_log(model_configs, args, log_path):
    """
    Saves a structured training log with all hyperparameters and results.
    This goes directly into the report.
    """
    with open(log_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("TRAINING LOG — CHARACTER-LEVEL NAME GENERATION\n")
        f.write("=" * 60 + "\n\n")

        f.write("SHARED HYPERPARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Embedding size:   {args.embed_size}\n")
        f.write(f"  Hidden size:      {args.hidden_size}\n")
        f.write(f"  Num layers:       {args.num_layers}\n")
        f.write(f"  Batch size:       {args.batch_size}\n")
        f.write(f"  Epochs:           {args.epochs}\n")
        f.write(f"  Learning rate:    {args.lr}\n")
        f.write(f"  Gradient clip:    {args.grad_clip}\n")
        f.write(f"  Optimizer:        Adam\n")
        f.write(f"  LR scheduler:    ReduceLROnPlateau (factor=0.5, patience=15)\n")
        f.write(f"  Loss function:    CrossEntropyLoss (ignore_index=PAD)\n\n")

        f.write("MODEL COMPARISON\n")
        f.write("-" * 40 + "\n")
        f.write(f"  {'Model':<25} {'Params':>10} {'Train Time':>12} {'Best Loss':>10}\n")
        f.write(f"  {'─'*25} {'─'*10} {'─'*12} {'─'*10}\n")

        for name, params, train_time, best_loss in model_configs:
            f.write(f"  {name:<25} {params:>10,} {train_time:>10.1f}s {best_loss:>10.4f}\n")

    print(f"  Training log saved to {log_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()

    # Create output directories
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)

    # Detect GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"CHARACTER-LEVEL NAME GENERATION — TRAINING")
    print(f"{'=' * 60}")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ---- Load data ----
    print(f"\n[1/5] Loading data...")
    names = load_names(args.data_file)
    vocab = CharVocab(names)
    print(f"  Vocabulary size: {vocab.size} characters")
    print(f"  Characters: {''.join(sorted(set(''.join(names))))}")

    dataset = NamesDataset(names, vocab)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # 0 for simplicity; increase if data loading is the bottleneck
        pin_memory=(device.type == "cuda"),
    )

    # ---- Create models ----
    print(f"\n[2/5] Creating models...")

    models = {
        "VanillaRNN": VanillaRNN(
            vocab_size=vocab.size,
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        ),
        "BidirectionalLSTM": BidirectionalLSTM(
            vocab_size=vocab.size,
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        ),
        "AttentionRNN": AttentionRNN(
            vocab_size=vocab.size,
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        ),
    }

    # Print architecture summaries
    for name, model in models.items():
        print_model_summary(model, name)

    # ---- Train each model ----
    print(f"\n[3/5] Training all models...")

    all_losses = []
    model_names = []
    model_configs = []

    for name, model in models.items():
        losses, train_time = train_model(
            model, name, train_loader, vocab, args, device
        )
        all_losses.append(losses)
        model_names.append(name)
        model_configs.append((name, model.count_parameters(), train_time, min(losses)))

    # ---- Plot loss curves ----
    print(f"\n[4/5] Plotting loss curves...")
    plot_path = os.path.join(args.output_dir, "loss_curves.png")
    plot_loss_curves(all_losses, model_names, plot_path)

    # ---- Save training log ----
    print(f"\n[5/5] Saving training log...")
    log_path = os.path.join(args.output_dir, "training_log.txt")
    save_training_log(model_configs, args, log_path)

    # ---- Final summary ----
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n  {'Model':<25} {'Params':>10} {'Time':>8} {'Loss':>8}")
    print(f"  {'─'*25} {'─'*10} {'─'*8} {'─'*8}")
    for name, params, t, loss in model_configs:
        print(f"  {name:<25} {params:>10,} {t:>6.0f}s {loss:>8.4f}")

    print(f"\n  All models and artifacts saved in {args.output_dir}/")
    print(f"\n  Next step: python generate_and_evaluate.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
