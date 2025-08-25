#@title GPT-3 Config Setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GPT3Config:
    def __init__(self):
        # Model architecture
        self.vocab_size = 50257     # Size of the GPT-3 tokenizer vocabulary
        self.d_model = 768          # Model hidden dimension (GPT-3 175B uses 12288)
        self.n_layers = 12          # Number of Transformer decoder layers (GPT-3 175B uses 96 layers)
        self.n_heads = 12           # Number of attention heads per layer (GPT-3 175B uses 96 heads)
        self.d_ff = 3072            # Feed-forward network hidden dimension (GPT-3 175B uses 49152)
        self.dropout = 0.1          # Dropout rate (GPT-3 paper did not use dropout)
        self.max_seq_len = 512      # Maximum sequence length (GPT-3 uses up to 2048 tokens)

        # Optimization / hyperparameters
        self.lr = 1e-4              # Learning rate for Adam optimizer
        self.betas = (0.9, 0.95)    # Beta values for Adam optimizer
        self.eps = 1e-8             # Epsilon for numerical stability in Adam optimizer
        self.weight_decay = 0.0     # Weight decay for regularization
        self.warmup_steps = 1000    # Number of steps to linearly warm up the LR
        self.lr_decay = "cosine"    # Learning rate decay schedule after warmup

        # FlashAttention / blocking
        self.block_size = 128             # Block size used by block-sparse
        self.use_flash_attention = True   # Flag to enable FlashAttention

        # Model details
        self.activation = "gelu"          # Activation function used in feed-forward layers
        self.initializer_range = 0.02     # Stddev for weight initialization

        # A100 optimization setup
        self.gradient_accumulation_steps = 16  # Large batch simulation
        self.mixed_precision = True            # FP16/BF16
        self.compile_model = True              # torch.compile
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Automatically select GPU if available, else use CPU
        self.epochs = 5                        # Number of epochs to train

config = GPT3Config()
