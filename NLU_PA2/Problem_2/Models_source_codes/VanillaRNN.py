# Model 1 — Vanilla RNN (from scratch, no nn.RNN)
#
# Architecture:
#   Embedding → Manual RNN cell (tanh) × num_layers → FC → logits
#   h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
#
# The manual RNN cell is implemented using nn.Linear weights
# to demonstrate understanding of the recurrent computation.

class VanillaRNN(nn.Module):
    """
    Vanilla RNN for character-level name generation.

    Manually implements the RNN recurrence:
        h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b_h)

    Hidden state is a list of tensors (one per layer), compatible
    with step-by-step autoregressive generation from the start.
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout=0.3):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.vocab_size  = vocab_size

        # Embedding: token index → dense vector
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        # Manual RNN weights per layer
        self.W_ih = nn.ModuleList()
        self.W_hh = nn.ModuleList()
        self.b_h  = nn.ParameterList()
        for layer in range(num_layers):
            in_dim = embed_dim if layer == 0 else hidden_size
            self.W_ih.append(nn.Linear(in_dim,     hidden_size, bias=False))
            self.W_hh.append(nn.Linear(hidden_size, hidden_size, bias=False))
            self.b_h.append(nn.Parameter(torch.zeros(hidden_size)))

        self.dropout = nn.Dropout(dropout)
        self.fc_out  = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        x      : (batch, seq_len)
        hidden : list of (batch, hidden_size) tensors, one per layer, or None
        Returns: logits (batch, seq_len, vocab_size), hidden (list)
        """
        batch_size, seq_len = x.shape
        embed = self.embedding(x)  # (batch, seq_len, embed_dim)

        # Initialise hidden states to zero if not provided
        if hidden is None:
            hidden = [torch.zeros(batch_size, self.hidden_size, device=x.device)
                      for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            x_t = embed[:, t, :]          # (batch, embed_dim)
            new_hidden = []
            for layer in range(self.num_layers):
                # Core RNN recurrence
                h_t = torch.tanh(
                    self.W_ih[layer](x_t) +
                    self.W_hh[layer](hidden[layer]) +
                    self.b_h[layer]
                )
                h_t = self.dropout(h_t)
                new_hidden.append(h_t)
                x_t = h_t              # output of this layer is input to next
            hidden = new_hidden
            outputs.append(hidden[-1].unsqueeze(1))

        all_out = torch.cat(outputs, dim=1)        # (batch, seq_len, hidden)
        logits  = self.fc_out(all_out)             # (batch, seq_len, vocab_size)
        return logits, hidden

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

rnn_model = VanillaRNN(VOCAB_SIZE, embed_dim=64, hidden_size=256, num_layers=2, dropout=0.3)
print(f"Model 1 — Vanilla RNN")
print(f"  embed_dim=64, hidden=256, layers=2, dropout=0.3")
print(f"  Trainable params: {rnn_model.count_parameters():,}")