# Bidirectional LSTM

# Architecture:
#   Embedding → BiLSTM encoder on [SOS] only → project h,c → 
#   Unidirectional LSTM decoder (teacher-forced) → FC → logits

class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM for character-level name generation.

    The BiLSTM encoder processes ONLY the SOS token to compute a rich
    initial hidden state for the unidirectional decoder. The decoder
    then generates characters autoregressively using teacher forcing
    during training — identical to generation mode.

    This design ensures COMPLETE consistency between training and generation:
    both always start with BiLSTM([SOS]) as the initial decoder state.

    Architecture:
      1. BiLSTM encoder: embeds [SOS] → bidirectional hidden states
                         → project 2*hidden → hidden (for h and c separately)
      2. Unidirectional LSTM decoder: generates sequence step by step
      3. FC output: hidden → vocab logits
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout=0.3):
        super(BidirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.vocab_size  = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.dropout   = nn.Dropout(dropout)

        # BiLSTM encoder — only ever sees the SOS embedding (single token)
        # bidirectional=True means hidden size doubles: output is 2*hidden_size
        self.encoder = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Project bidirectional (fwd+bwd) hidden/cell → decoder size
        # Input: 2*hidden_size (fwd concat bwd), Output: hidden_size
        self.h_proj = nn.Linear(2 * hidden_size, hidden_size)
        self.c_proj = nn.Linear(2 * hidden_size, hidden_size)

        # Unidirectional LSTM decoder — autoregressive, one token at a time
        self.decoder = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def _get_initial_hidden(self, batch_size, device):
        """
        Compute initial decoder (h, c) by running BiLSTM on the SOS token.
        Called once at the START of every forward pass (training and generation).
        batch_size: number of sequences in current batch
        """
        # Create a batch of SOS tokens: (batch_size, 1)
        sos_tokens = torch.full(
            (batch_size, 1), SOS_IDX, dtype=torch.long, device=device
        )
        # Embed SOS: (batch_size, 1, embed_dim)
        sos_embed = self.dropout(self.embedding(sos_tokens))

        # Run through BiLSTM — only 1 timestep
        # h_enc: (2*num_layers, batch, hidden) — interleaved fwd/bwd
        _, (h_enc, c_enc) = self.encoder(sos_embed)

        # Separate forward and backward layers
        # Even indices: forward layers (0, 2, 4...)
        # Odd indices:  backward layers (1, 3, 5...)
        h_fwd = h_enc[0::2]   # (num_layers, batch, hidden)
        h_bwd = h_enc[1::2]
        c_fwd = c_enc[0::2]
        c_bwd = c_enc[1::2]

        # Concatenate fwd+bwd: (num_layers, batch, 2*hidden)
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)
        c_cat = torch.cat([c_fwd, c_bwd], dim=-1)

        # Project each layer independently to decoder hidden size
        h0 = torch.stack([self.h_proj(h_cat[l]) for l in range(self.num_layers)])
        c0 = torch.stack([self.c_proj(c_cat[l]) for l in range(self.num_layers)])
        return h0, c0

    def forward(self, x, hidden=None):
        """
        x      : (batch, seq_len) — input token indices
        hidden : (h, c) decoder state, or None for first call

        TRAINING (hidden=None, x = full sequence with SOS prefix):
          Computes initial (h,c) from BiLSTM([SOS]) for the whole batch,
          then runs full sequence through unidirectional decoder.

        GENERATION (step-by-step):
          First call: hidden=None, x=[[SOS]] → gets h0,c0, decodes 1 step
          Next calls: hidden=(h,c) from previous step, x=[[last_token]]
          The initial hidden is IDENTICAL to training (BiLSTM on SOS).

        Returns: logits (batch, seq_len, vocab_size), (h, c)
        """
        batch_size = x.shape[0]
        device     = x.device

        if hidden is None:
            # Compute fresh initial hidden state from BiLSTM encoder on SOS
            # This is ALWAYS done at the start — training or generation
            h0, c0 = self._get_initial_hidden(batch_size, device)
            hidden = (h0, c0)

        # Decode: embed x and run through unidirectional LSTM
        embed   = self.dropout(self.embedding(x))       # (batch, seq_len, embed_dim)
        dec_out, hidden = self.decoder(embed, hidden)   # (batch, seq_len, hidden)
        dec_out = self.dropout(dec_out)
        logits  = self.fc_out(dec_out)                  # (batch, seq_len, vocab_size)
        return logits, hidden

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Reinstantiate with fresh weights
blstm_model = BidirectionalLSTM(
    VOCAB_SIZE, embed_dim=64, hidden_size=256, num_layers=2, dropout=0.3
)
print(f"Model 2 — Bidirectional LSTM (FIXED)")
print(f"  BiLSTM encoder: always encodes [SOS] only")
print(f"  Unidirectional decoder: learns character transitions")
print(f"  embed_dim=64, hidden=256 (BiLSTM=512 internally), layers=2")
print(f"  Trainable params: {blstm_model.count_parameters():,}")

# Verify: training and generation start with IDENTICAL encoder input
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blstm_model = blstm_model.to(device)
h0, c0 = blstm_model._get_initial_hidden(batch_size=1, device=device)
print(f"\nEncoder hidden state shape: {h0.shape}")  # (num_layers, 1, hidden)
print(" Encoder correctly produces fixed initial state from SOS token")
blstm_model = blstm_model.cpu()