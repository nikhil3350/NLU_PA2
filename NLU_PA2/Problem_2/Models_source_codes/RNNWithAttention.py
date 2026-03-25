# Model 3 — RNN with Basic Bahdanau Attention Mechanism
#
# Architecture:
#   Embedding → RNN (step-by-step) → Bahdanau Attention over past GRU outputs
#             → gated blend(h_t, context) → FC → logits
#
# Attention formula:
#   score(h_t, h_i) = v^T * tanh(W_q * h_t + W_k * h_i)
#   weights = softmax(scores)
#   context = Σ weights_i * h_i
#
# KEY DESIGN DECISION for autoregressive consistency:
#   Attention attends over the GRU's OWN past outputs [h_0..h_{t-1}].
#   Training: keys built from earlier timesteps in same batch.
#   Generation: keys are a growing buffer of past outputs passed explicitly.
#   → IDENTICAL computation in both modes.

class BahdanauAttention(nn.Module):
    """
    Basic Bahdanau (additive) attention.
    score(query, key_i) = v^T * tanh(W_q * query + W_k * key_i)
    """
    def __init__(self, hidden_size, attention_dim):
        super(BahdanauAttention, self).__init__()
        self.W_q = nn.Linear(hidden_size, attention_dim, bias=False)
        self.W_k = nn.Linear(hidden_size, attention_dim, bias=False)
        self.v   = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, query, keys):
        """
        query : (batch, hidden)    current GRU output
        keys  : (batch, t, hidden) past GRU outputs
        Returns context (batch, hidden), weights (batch, t)
        """
        q       = self.W_q(query).unsqueeze(1)                  # (batch, 1, attn_dim)
        k       = self.W_k(keys)                                # (batch, t, attn_dim)
        scores  = self.v(torch.tanh(q + k)).squeeze(-1)         # (batch, t)
        weights = F.softmax(scores, dim=-1)                     # (batch, t)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)  # (batch, hidden)
        return context, weights


class RNNWithAttention(nn.Module):
    """
    RNN with Basic Bahdanau Attention for character-level name generation.

    At each timestep t:
      1. RNN processes embedding of current token → h_t
      2. Attention over past GRU outputs [h_0..h_{t-1}] → context c_t
      3. Gate blends h_t and c_t → output representation o_t
      4. FC(o_t) → logits over vocabulary

    At t=0: no past outputs exist, so output = FC(dropout(h_0)) directly.

    past_outputs parameter:
      - Training: None (keys computed internally from current batch)
      - Generation: growing tensor of past h's, passed from outside
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers,
                 attention_dim=128, dropout=0.3):
        super(RNNWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.vocab_size  = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.dropout   = nn.Dropout(dropout)

        # GRU: gated recurrence — no vanishing gradient, standard for attention
        self.gru = nn.GRU(
            input_size=embed_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Bahdanau attention module
        self.attention = BahdanauAttention(hidden_size, attention_dim)

        # Gating: learned blend of current hidden + attended context
        self.gate   = nn.Linear(2 * hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None, past_outputs=None):
        """
        x            : (batch, seq_len)
        hidden       : GRU hidden (num_layers, batch, hidden) or None
        past_outputs : (batch, T_past, hidden) for generation mode, else None
        Returns: logits (batch, seq_len, vocab), hidden, gru_outputs
        """
        embed = self.dropout(self.embedding(x))
        gru_out, hidden = self.gru(embed, hidden)  # (batch, seq_len, hidden)

        all_logits = []
        for t in range(x.shape[1]):
            h_t = gru_out[:, t, :]                 # (batch, hidden)

            if past_outputs is not None:
                # Generation mode: use explicit past buffer
                keys = past_outputs
            else:
                # Training mode: causal — only attend to positions before t
                keys = gru_out[:, :t, :]

            if keys.shape[1] == 0:
                # t=0: no past context yet
                out = self.dropout(h_t)
            else:
                context, _ = self.attention(h_t, keys)
                # Gated blend
                out = torch.relu(self.gate(torch.cat([h_t, context], dim=-1)))
                out = self.dropout(out)

            all_logits.append(self.fc_out(out).unsqueeze(1))

        logits = torch.cat(all_logits, dim=1)      # (batch, seq_len, vocab)
        return logits, hidden, gru_out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


attn_model = RNNWithAttention(
    VOCAB_SIZE, embed_dim=64, hidden_size=256,
    num_layers=2, attention_dim=128, dropout=0.3
)
print(f"Model 3 — RNN with Bahdanau Attention")
print(f"  embed_dim=64, hidden=256, layers=2, attn_dim=128, dropout=0.3")
print(f"  Trainable params: {attn_model.count_parameters():,}")