"""
=============================================================================
MODELS: Character-Level Name Generation Architectures
=============================================================================
Implements three recurrent architectures FROM SCRATCH for character-level
name generation. "From scratch" means we implement the recurrent cells
manually using basic PyTorch operations (nn.Linear, torch.tanh, torch.sigmoid)
rather than using nn.RNN, nn.LSTM, or nn.GRU.

Models:
  1. VanillaRNN        — basic Elman RNN
  2. BidirectionalLSTM — forward + backward LSTM, concat hidden states
  3. AttentionRNN      — RNN encoder + Bahdanau attention decoder

All models follow the same interface:
  - forward(x, lengths)  → logits for training
  - generate(vocab, ...)  → list of generated name strings
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# MODEL 1: VANILLA RNN (Elman Network)
# ============================================================================

class VanillaRNNCell(nn.Module):
    """
    A single RNN cell implemented from scratch.

    Computation at each timestep t:
        h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)

    Where:
        x_t    = input at time t (embedding vector)
        h_{t-1}= previous hidden state
        W_ih   = input-to-hidden weight matrix
        W_hh   = hidden-to-hidden weight matrix (recurrent weights)
        b_ih, b_hh = bias vectors

    This is the fundamental recurrent unit — the hidden state h_t
    carries information from all previous timesteps.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Input-to-hidden transformation
        self.W_ih = nn.Linear(input_size, hidden_size)

        # Hidden-to-hidden transformation (the recurrent connection)
        self.W_hh = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h_prev):
        """
        Args:
            x:      (batch_size, input_size) — current input
            h_prev: (batch_size, hidden_size) — previous hidden state
        Returns:
            h_new:  (batch_size, hidden_size) — new hidden state
        """
        # Core RNN equation: h_t = tanh(W_ih * x_t + W_hh * h_{t-1})
        h_new = torch.tanh(self.W_ih(x) + self.W_hh(h_prev))
        return h_new


class VanillaRNN(nn.Module):
    """
    Vanilla RNN for character-level name generation.

    Architecture:
        Character → Embedding → RNN Cell (unrolled over time) → FC → Logits

    The model processes one character at a time. At each step, the RNN cell
    updates its hidden state based on the current character embedding and
    the previous hidden state. The hidden state is then projected to
    vocabulary-sized logits to predict the next character.
    """

    def __init__(self, vocab_size, embed_size=64, hidden_size=128, num_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Character embedding layer: maps char index → dense vector
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Stack of RNN cells (multiple layers for depth)
        # Layer 0 takes embedding input; layers 1+ take previous layer's hidden
        self.rnn_cells = nn.ModuleList()
        for layer in range(num_layers):
            input_dim = embed_size if layer == 0 else hidden_size
            self.rnn_cells.append(VanillaRNNCell(input_dim, hidden_size))

        # Output projection: hidden state → character logits
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, lengths=None):
        """
        Forward pass with teacher forcing (training mode).

        Args:
            x:       (batch_size, seq_len) — input character indices
            lengths: (batch_size,) — actual lengths (unused here, for API compat)
        Returns:
            logits:  (batch_size, seq_len, vocab_size) — next-char predictions
        """
        batch_size, seq_len = x.shape

        # Embed input characters
        embedded = self.embedding(x)  # (batch, seq_len, embed_size)

        # Initialize hidden states to zeros for each layer
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device)
             for _ in range(self.num_layers)]

        # Process each timestep sequentially
        outputs = []
        for t in range(seq_len):
            inp = embedded[:, t, :]  # (batch, embed_size)

            # Pass through each RNN layer
            for layer in range(self.num_layers):
                h[layer] = self.rnn_cells[layer](inp, h[layer])
                inp = h[layer]  # output of this layer → input to next

            # Project top layer's hidden state to logits
            logit = self.fc_out(h[-1])  # (batch, vocab_size)
            outputs.append(logit)

        # Stack timestep outputs → (batch, seq_len, vocab_size)
        logits = torch.stack(outputs, dim=1)
        return logits

    def generate(self, vocab, max_len=20, num_names=100, temperature=0.8, device="cpu"):
        """
        Generate names by sampling one character at a time.

        Starts with SOS token, samples next character from the predicted
        distribution, feeds it back as input, repeats until EOS or max_len.

        Args:
            vocab:       CharVocab instance
            max_len:     maximum name length
            num_names:   how many names to generate
            temperature: sampling temperature (lower = more conservative)
            device:      cpu or cuda

        Returns:
            list of generated name strings
        """
        self.eval()
        generated_names = []

        with torch.no_grad():
            for _ in range(num_names):
                # Start with SOS token
                current_char = torch.tensor([[vocab.sos_idx]], device=device)
                h = [torch.zeros(1, self.hidden_size, device=device)
                     for _ in range(self.num_layers)]

                name_indices = []

                for _ in range(max_len):
                    emb = self.embedding(current_char).squeeze(1)  # (1, embed_size)

                    # Forward through all RNN layers
                    inp = emb
                    for layer in range(self.num_layers):
                        h[layer] = self.rnn_cells[layer](inp, h[layer])
                        inp = h[layer]

                    # Get next character distribution
                    logits = self.fc_out(h[-1]) / temperature
                    probs = F.softmax(logits, dim=-1)

                    # Sample from the distribution
                    next_char_idx = torch.multinomial(probs, 1).item()

                    # Stop if EOS
                    if next_char_idx == vocab.eos_idx:
                        break

                    name_indices.append(next_char_idx)
                    current_char = torch.tensor([[next_char_idx]], device=device)

                # Decode indices back to string
                name = vocab.decode(name_indices)
                if name:  # skip empty names
                    generated_names.append(name)

        self.train()
        return generated_names

    def count_parameters(self):
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# MODEL 2: BIDIRECTIONAL LSTM
# ============================================================================

class LSTMCell(nn.Module):
    """
    A single LSTM cell implemented from scratch.

    The LSTM uses gating mechanisms to control information flow,
    solving the vanishing gradient problem of vanilla RNNs.

    Gate equations:
        f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)   — forget gate
        i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)   — input gate
        g_t = tanh(W_g @ [h_{t-1}, x_t] + b_g)      — candidate cell state
        o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)    — output gate

    State updates:
        c_t = f_t * c_{t-1} + i_t * g_t              — new cell state
        h_t = o_t * tanh(c_t)                          — new hidden state

    Where:
        f_t = forget gate (what to discard from cell state)
        i_t = input gate (what new info to store)
        g_t = candidate values to add to cell state
        o_t = output gate (what part of cell state to output)
        c_t = cell state (long-term memory)
        h_t = hidden state (short-term output)
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Combined linear transformation for all 4 gates (more efficient)
        # Input: concatenation of [x_t, h_{t-1}]
        # Output: 4 * hidden_size (one chunk per gate)
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, h_prev, c_prev):
        """
        Args:
            x:      (batch_size, input_size) — current input
            h_prev: (batch_size, hidden_size) — previous hidden state
            c_prev: (batch_size, hidden_size) — previous cell state
        Returns:
            h_new:  (batch_size, hidden_size) — new hidden state
            c_new:  (batch_size, hidden_size) — new cell state
        """
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)  # (batch, input+hidden)

        # Compute all four gates in one matrix multiplication
        gate_values = self.gates(combined)  # (batch, 4*hidden)

        # Split into individual gates
        i_gate, f_gate, g_gate, o_gate = gate_values.chunk(4, dim=1)

        # Apply activations
        i_gate = torch.sigmoid(i_gate)   # input gate
        f_gate = torch.sigmoid(f_gate)   # forget gate
        g_gate = torch.tanh(g_gate)      # candidate cell state
        o_gate = torch.sigmoid(o_gate)   # output gate

        # Update cell state: forget old info + add new info
        c_new = f_gate * c_prev + i_gate * g_gate

        # Compute new hidden state from cell state
        h_new = o_gate * torch.tanh(c_new)

        return h_new, c_new


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM for character-level name generation.

    Architecture:
        Char → Embedding → Forward LSTM + Backward LSTM → Concat → FC → Logits

    The forward LSTM reads left-to-right, the backward LSTM reads right-to-left.
    Their hidden states are concatenated at each timestep, giving the model
    context from both directions.

    IMPORTANT NOTE ON GENERATION:
    Bidirectional models are not naturally suited for autoregressive generation
    because during generation we don't have future context. For generation,
    we use only the forward direction. This limitation is discussed in
    the qualitative analysis (Task 3).
    """

    def __init__(self, vocab_size, embed_size=64, hidden_size=128, num_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Forward LSTM cells (left → right)
        self.fwd_cells = nn.ModuleList()
        for layer in range(num_layers):
            input_dim = embed_size if layer == 0 else hidden_size
            self.fwd_cells.append(LSTMCell(input_dim, hidden_size))

        # Backward LSTM cells (right → left)
        self.bwd_cells = nn.ModuleList()
        for layer in range(num_layers):
            input_dim = embed_size if layer == 0 else hidden_size
            self.bwd_cells.append(LSTMCell(input_dim, hidden_size))

        # Output projection: concat(forward_h, backward_h) → logits
        # Factor of 2 because we concatenate forward and backward hidden states
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, lengths=None):
        """
        Forward pass processing the sequence in both directions.

        Args:
            x:       (batch_size, seq_len) — input character indices
            lengths: (batch_size,) — actual sequence lengths (for masking)
        Returns:
            logits:  (batch_size, seq_len, vocab_size) — next-char predictions
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Embed input characters
        embedded = self.embedding(x)  # (batch, seq_len, embed_size)

        # ---- FORWARD PASS (left → right) ----
        fwd_h = [torch.zeros(batch_size, self.hidden_size, device=device)
                 for _ in range(self.num_layers)]
        fwd_c = [torch.zeros(batch_size, self.hidden_size, device=device)
                 for _ in range(self.num_layers)]

        fwd_outputs = []
        for t in range(seq_len):
            inp = embedded[:, t, :]
            for layer in range(self.num_layers):
                fwd_h[layer], fwd_c[layer] = self.fwd_cells[layer](
                    inp, fwd_h[layer], fwd_c[layer]
                )
                inp = fwd_h[layer]
                # For multi-layer, we need to handle the concatenation
                # But at intermediate layers within one direction,
                # we pass just the hidden state
            fwd_outputs.append(fwd_h[-1])

        fwd_outputs = torch.stack(fwd_outputs, dim=1)  # (batch, seq_len, hidden)

        # ---- BACKWARD PASS (right → left) ----
        bwd_h = [torch.zeros(batch_size, self.hidden_size, device=device)
                 for _ in range(self.num_layers)]
        bwd_c = [torch.zeros(batch_size, self.hidden_size, device=device)
                 for _ in range(self.num_layers)]

        bwd_outputs = []
        for t in range(seq_len - 1, -1, -1):  # iterate in reverse
            inp = embedded[:, t, :]
            for layer in range(self.num_layers):
                bwd_h[layer], bwd_c[layer] = self.bwd_cells[layer](
                    inp, bwd_h[layer], bwd_c[layer]
                )
                inp = bwd_h[layer]
            bwd_outputs.append(bwd_h[-1])

        # Reverse backward outputs to align with forward outputs temporally
        bwd_outputs = list(reversed(bwd_outputs))
        bwd_outputs = torch.stack(bwd_outputs, dim=1)  # (batch, seq_len, hidden)

        # ---- CONCATENATE FORWARD + BACKWARD ----
        combined = torch.cat([fwd_outputs, bwd_outputs], dim=2)
        # Shape: (batch, seq_len, hidden * 2)

        # Project to vocabulary logits
        logits = self.fc_out(combined)  # (batch, seq_len, vocab_size)
        return logits

    def generate(self, vocab, max_len=20, num_names=100, temperature=0.8, device="cpu"):
        """
        Generate names using ONLY the forward LSTM direction.

        Since we can't look ahead during generation (autoregressive),
        the backward LSTM is not used here. This is a known limitation
        of bidirectional models for generation tasks — we train with
        bidirectional context but generate with unidirectional context.

        This asymmetry between training and generation is a key
        discussion point for the report.
        """
        self.eval()
        generated_names = []

        with torch.no_grad():
            for _ in range(num_names):
                current_char = torch.tensor([[vocab.sos_idx]], device=device)

                fwd_h = [torch.zeros(1, self.hidden_size, device=device)
                         for _ in range(self.num_layers)]
                fwd_c = [torch.zeros(1, self.hidden_size, device=device)
                         for _ in range(self.num_layers)]

                name_indices = []

                for _ in range(max_len):
                    emb = self.embedding(current_char).squeeze(1)

                    inp = emb
                    for layer in range(self.num_layers):
                        fwd_h[layer], fwd_c[layer] = self.fwd_cells[layer](
                            inp, fwd_h[layer], fwd_c[layer]
                        )
                        inp = fwd_h[layer]

                    # Use only forward hidden + zero backward (no future context)
                    # This creates a mismatch with training but is the only option
                    zero_bwd = torch.zeros(1, self.hidden_size, device=device)
                    combined_h = torch.cat([fwd_h[-1], zero_bwd], dim=1)

                    logits = self.fc_out(combined_h) / temperature
                    probs = F.softmax(logits, dim=-1)
                    next_char_idx = torch.multinomial(probs, 1).item()

                    if next_char_idx == vocab.eos_idx:
                        break

                    name_indices.append(next_char_idx)
                    current_char = torch.tensor([[next_char_idx]], device=device)

                name = vocab.decode(name_indices)
                if name:
                    generated_names.append(name)

        self.train()
        return generated_names

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# MODEL 3: RNN WITH BASIC ATTENTION MECHANISM
# ============================================================================

class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention mechanism implemented from scratch.

    At each decoding step t, attention computes a context vector by
    looking back at all encoder hidden states and assigning weights
    based on relevance.

    Steps:
        1. Score each encoder hidden state: score_i = V @ tanh(W_h @ h_i + W_s @ s_t)
        2. Normalize scores to get attention weights: alpha = softmax(scores)
        3. Compute context vector: context = sum(alpha_i * h_i)

    Where:
        h_i = encoder hidden state at position i
        s_t = current decoder hidden state
        V, W_h, W_s = learnable parameters

    This allows the model to "attend" to different parts of the name
    generated so far, helping it maintain coherence over longer sequences.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Attention scoring parameters
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)  # encoder projection
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)  # decoder projection
        self.V = nn.Linear(hidden_size, 1, bias=False)              # score reduction

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Args:
            decoder_hidden:  (batch, hidden) — current decoder state
            encoder_outputs: (batch, src_len, hidden) — all encoder states
            mask:            (batch, src_len) — True for valid positions

        Returns:
            context: (batch, hidden) — weighted sum of encoder outputs
            weights: (batch, src_len) — attention weights (for visualization)
        """
        src_len = encoder_outputs.size(1)

        # Expand decoder hidden to match encoder sequence length
        # (batch, hidden) → (batch, src_len, hidden)
        decoder_expanded = decoder_hidden.unsqueeze(1).expand(-1, src_len, -1)

        # Compute attention scores using additive (Bahdanau) scoring
        # score_i = V * tanh(W_h * h_i + W_s * s_t)
        energy = torch.tanh(
            self.W_h(encoder_outputs) + self.W_s(decoder_expanded)
        )  # (batch, src_len, hidden)

        scores = self.V(energy).squeeze(2)  # (batch, src_len)

        # Apply mask: set scores for padding positions to -inf
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        # Normalize scores to attention weights
        weights = F.softmax(scores, dim=1)  # (batch, src_len)

        # Compute context vector as weighted sum of encoder outputs
        # (batch, 1, src_len) @ (batch, src_len, hidden) → (batch, 1, hidden)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, weights


class AttentionRNN(nn.Module):
    """
    RNN with Bahdanau Attention for character-level name generation.

    Architecture:
        1. ENCODER: RNN processes input sequence, produces hidden states at each step
        2. ATTENTION: At each step, attends over all previous hidden states
        3. DECODER: Combines attention context + current input to predict next char

    This is a self-attentive autoregressive model — at each timestep t, it:
        a) Runs the RNN cell to get h_t
        b) Attends over h_1..h_{t-1} to get context_t
        c) Combines [h_t; context_t] to predict the next character

    The attention mechanism helps the model look back at patterns in the
    name generated so far, which is useful for maintaining phonetic
    consistency and capturing character-level dependencies.
    """

    def __init__(self, vocab_size, embed_size=64, hidden_size=128, num_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Encoder RNN: processes input and produces hidden states
        # Using our from-scratch LSTM cell for the encoder
        self.encoder_cells = nn.ModuleList()
        for layer in range(num_layers):
            input_dim = embed_size if layer == 0 else hidden_size
            self.encoder_cells.append(LSTMCell(input_dim, hidden_size))

        # Attention mechanism
        self.attention = BahdanauAttention(hidden_size)

        # Combine attention context + hidden state for final prediction
        # Input: [hidden_state; attention_context] → output logits
        self.fc_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, lengths=None):
        """
        Forward pass with self-attention over encoder hidden states.

        At each timestep t, the model:
          1. Computes encoder hidden state h_t from input
          2. Attends over all hidden states h_1..h_{t-1}
          3. Combines context and h_t to predict next character

        For the first timestep, there's nothing to attend to, so
        context is a zero vector.
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Embed input characters
        embedded = self.embedding(x)  # (batch, seq_len, embed_size)

        # Initialize hidden and cell states
        h = [torch.zeros(batch_size, self.hidden_size, device=device)
             for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size, device=device)
             for _ in range(self.num_layers)]

        # Storage for all encoder hidden states (for attention)
        all_hidden = []

        # Create mask for attention (True = valid position)
        # Build incrementally as we process each timestep
        if lengths is not None:
            # Full mask based on actual lengths
            full_mask = torch.arange(seq_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        else:
            full_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        outputs = []

        for t in range(seq_len):
            inp = embedded[:, t, :]

            # Run through encoder LSTM layers
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.encoder_cells[layer](inp, h[layer], c[layer])
                inp = h[layer]

            # Store this timestep's hidden state
            all_hidden.append(h[-1])

            # Compute attention over ALL hidden states so far
            if t > 0:
                # Stack all previous hidden states: (batch, t, hidden)
                encoder_states = torch.stack(all_hidden[:t], dim=1)

                # Attention mask: only attend to positions 0..t-1
                attn_mask = full_mask[:, :t]

                # Get context from attention
                context, _ = self.attention(h[-1], encoder_states, attn_mask)
            else:
                # First timestep: no previous states to attend to
                context = torch.zeros(batch_size, self.hidden_size, device=device)

            # Combine hidden state and attention context
            combined = torch.cat([h[-1], context], dim=1)  # (batch, hidden*2)
            combined = torch.tanh(self.fc_combine(combined))  # (batch, hidden)

            # Project to logits
            logit = self.fc_out(combined)  # (batch, vocab_size)
            outputs.append(logit)

        logits = torch.stack(outputs, dim=1)  # (batch, seq_len, vocab_size)
        return logits

    def generate(self, vocab, max_len=20, num_names=100, temperature=0.8, device="cpu"):
        """
        Generate names with attention-guided sampling.

        At each step, the model attends over all characters generated so far,
        allowing it to maintain consistency (e.g., if it started with a pattern,
        attention helps it continue that pattern).
        """
        self.eval()
        generated_names = []

        with torch.no_grad():
            for _ in range(num_names):
                current_char = torch.tensor([[vocab.sos_idx]], device=device)

                h = [torch.zeros(1, self.hidden_size, device=device)
                     for _ in range(self.num_layers)]
                c = [torch.zeros(1, self.hidden_size, device=device)
                     for _ in range(self.num_layers)]

                all_hidden = []
                name_indices = []

                for step in range(max_len):
                    emb = self.embedding(current_char).squeeze(1)

                    inp = emb
                    for layer in range(self.num_layers):
                        h[layer], c[layer] = self.encoder_cells[layer](
                            inp, h[layer], c[layer]
                        )
                        inp = h[layer]

                    all_hidden.append(h[-1])

                    # Attend over previous hidden states
                    if step > 0:
                        encoder_states = torch.stack(all_hidden[:step], dim=1)
                        context, _ = self.attention(h[-1], encoder_states)
                    else:
                        context = torch.zeros(1, self.hidden_size, device=device)

                    combined = torch.cat([h[-1], context], dim=1)
                    combined = torch.tanh(self.fc_combine(combined))

                    logits = self.fc_out(combined) / temperature
                    probs = F.softmax(logits, dim=-1)
                    next_char_idx = torch.multinomial(probs, 1).item()

                    if next_char_idx == vocab.eos_idx:
                        break

                    name_indices.append(next_char_idx)
                    current_char = torch.tensor([[next_char_idx]], device=device)

                name = vocab.decode(name_indices)
                if name:
                    generated_names.append(name)

        self.train()
        return generated_names

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# MODEL SUMMARY UTILITY
# ============================================================================

def print_model_summary(model, model_name):
    """
    Prints a structured summary of model architecture and parameter counts.
    Useful for the report (Task 1 requirement).
    """
    total_params = model.count_parameters()

    print(f"\n{'=' * 50}")
    print(f"  {model_name}")
    print(f"{'=' * 50}")
    print(f"  Total trainable parameters: {total_params:,}")
    print(f"\n  Layer breakdown:")
    print(f"  {'Layer':<35} {'Params':>10}")
    print(f"  {'-'*35} {'-'*10}")

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name:<35} {param.numel():>10,}")

    print(f"  {'-'*35} {'-'*10}")
    print(f"  {'TOTAL':<35} {total_params:>10,}")
    print()
