"""
=============================================================================
DATASET: Character-Level Name Dataset
=============================================================================
Handles loading names, building character vocabulary, and encoding/decoding
sequences for character-level generation.

Special tokens:
  SOS (^) — start of sequence, prepended to every name during training
  EOS ($) — end of sequence, appended to every name during training

Example:
  "Aarav" → ['^', 'A', 'a', 'r', 'a', 'v', '$']
  Encoded → [1, 28, 2, 19, 2, 23, 0]
=============================================================================
"""

import torch
from torch.utils.data import Dataset
import os


# Special token characters
PAD_TOKEN = "_"   # padding (index 0, not used in generation)
SOS_TOKEN = "^"   # start of sequence
EOS_TOKEN = "$"   # end of sequence


class CharVocab:
    """
    Character-level vocabulary that maps characters to indices and back.

    Built from the training names file. Includes special tokens for
    sequence boundaries (SOS, EOS) and padding.
    """

    def __init__(self, names):
        """
        Builds vocabulary from a list of name strings.
        Collects all unique characters and assigns each an integer index.
        """
        # Collect all unique characters across all names
        chars = set()
        for name in names:
            chars.update(name)

        # Sort for deterministic ordering across runs
        chars = sorted(chars)

        # Build char → index mapping
        # Reserve first 3 indices for special tokens
        self.char_to_idx = {
            PAD_TOKEN: 0,
            SOS_TOKEN: 1,
            EOS_TOKEN: 2,
        }

        # Add all regular characters starting from index 3
        for i, ch in enumerate(chars):
            self.char_to_idx[ch] = i + 3

        # Build reverse mapping: index → char
        self.idx_to_char = {idx: ch for ch, idx in self.char_to_idx.items()}

        # Store special token indices for easy access
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2

        # Vocabulary size (total unique tokens including specials)
        self.size = len(self.char_to_idx)

    def encode(self, name):
        """
        Converts a name string to a list of token indices.
        Prepends SOS and appends EOS.
        Example: "Aarav" → [1, A_idx, a_idx, r_idx, a_idx, v_idx, 2]
        """
        indices = [self.sos_idx]
        for ch in name:
            indices.append(self.char_to_idx[ch])
        indices.append(self.eos_idx)
        return indices

    def decode(self, indices):
        """
        Converts a list of token indices back to a string.
        Stops at the first EOS token and strips SOS/EOS/PAD.
        """
        chars = []
        for idx in indices:
            if idx == self.eos_idx:
                break
            if idx == self.sos_idx or idx == self.pad_idx:
                continue
            chars.append(self.idx_to_char.get(idx, "?"))
        return "".join(chars)

    def __len__(self):
        return self.size


class NamesDataset(Dataset):
    """
    PyTorch Dataset for character-level name generation.

    Each sample is a single name encoded as a sequence of character indices.
    For training with teacher forcing:
      - input  = [SOS, c1, c2, ..., cn]      (all chars except last)
      - target = [c1, c2, ..., cn, EOS]       (all chars except first)

    The model learns to predict the next character given the previous ones.
    """

    def __init__(self, names, vocab):
        """
        Args:
            names: list of name strings
            vocab: CharVocab instance
        """
        self.names = names
        self.vocab = vocab

        # Pre-encode all names for faster training
        self.encoded = [vocab.encode(name) for name in names]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        """
        Returns (input_tensor, target_tensor) for one name.

        For name "Aarav" encoded as [SOS, A, a, r, a, v, EOS]:
          input  = [SOS, A, a, r, a, v]     → what the model sees
          target = [A, a, r, a, v, EOS]     → what the model should predict
        """
        encoded = self.encoded[idx]
        input_seq = torch.tensor(encoded[:-1], dtype=torch.long)
        target_seq = torch.tensor(encoded[1:], dtype=torch.long)
        return input_seq, target_seq


def load_names(filepath="TrainingNames.txt"):
    """
    Loads names from a text file (one name per line).
    Strips whitespace and filters out empty lines.
    """
    with open(filepath, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    print(f"  Loaded {len(names)} names from {filepath}")
    print(f"  Shortest: {min(len(n) for n in names)} chars, "
          f"Longest: {max(len(n) for n in names)} chars")
    return names


def collate_fn(batch):
    """
    Custom collate function that pads sequences to the same length
    within a batch. Required because names have different lengths.

    Pads with PAD_TOKEN (index 0) on the right side.
    Returns:
      - inputs:  (batch_size, max_seq_len) padded input tensor
      - targets: (batch_size, max_seq_len) padded target tensor
      - lengths: (batch_size,) original lengths (before padding)
    """
    inputs, targets = zip(*batch)

    # Find max length in this batch
    lengths = [len(inp) for inp in inputs]
    max_len = max(lengths)

    # Pad sequences to max_len
    padded_inputs = torch.zeros(len(inputs), max_len, dtype=torch.long)
    padded_targets = torch.zeros(len(targets), max_len, dtype=torch.long)

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        padded_inputs[i, :len(inp)] = inp
        padded_targets[i, :len(tgt)] = tgt

    lengths = torch.tensor(lengths, dtype=torch.long)
    return padded_inputs, padded_targets, lengths
