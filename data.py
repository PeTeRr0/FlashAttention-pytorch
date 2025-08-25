#@title GPT-3 Dataset Preparation

class GPT3Dataset(Dataset):
    def __init__(self, texts, tokenizer, max_seq_len, vocab_size=None, sequential=True):
        """
        Args:
            texts (list[str]): Raw text samples
            tokenizer: GPT tokenizer (e.g., tiktoken)
            max_seq_len (int): Max sequence length
            vocab_size (int, optional): Limit vocab to config size
            sequential (bool):
                - True: sequential slicing of tokens (deterministic)
                - False: random subsequence sampling (better generalization)
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.sequential = sequential

        # Tokenize all texts once into one long stream
        print("Tokenizing dataset...")
        start_time = time.time()

        self.tokens = []
        for i, text in enumerate(texts):
            # encode returns list[int] token ids
            token_ids = tokenizer.encode(text)
            self.tokens.extend(token_ids)

            # Progress logging every 10k texts helps estimate runtime on large corpora
            if (i + 1) % 10000 == 0:
                elapsed = (time.time() - start_time) / 60
                progress = (i + 1) / len(texts) * 100
                print(f"Progress: {i+1:,}/{len(texts):,} ({progress:.1f}%) - Elapsed: {elapsed:.1f}min")

        # Tokenization completed
        total_time = (time.time() - start_time) / 60
        print(f"Tokenization completed in {total_time:.1f} minutes")

        # store stats used by __len__ and __getitem__
        self.total_tokens = len(self.tokens)
        # Number of full sequences
        self.num_sequences = self.total_tokens // self.max_seq_len
        print(f"Total tokens: {self.total_tokens:,}")
        print(f"Total usable sequences: {self.num_sequences:,}")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Determine start index for the requested sequence
        if self.sequential:
            start = idx * self.max_seq_len
        else:
            # random offset for more variety
            start = random.randint(0, self.total_tokens - self.max_seq_len - 1)

        end = start + self.max_seq_len
        seq = self.tokens[start:end]

        # Ensure fixed length by padding with 0's
        if len(seq) < self.max_seq_len:
            seq += [0] * (self.max_seq_len - len(seq))

        # Clamp IDs to vocab_size
        seq = [min(t, self.vocab_size - 1) for t in seq]

        # Inputs are tokens[:-1], targets are tokens[1:] (next-token prediction)
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)
        return input_ids, target_ids


# --- Load dataset (OpenWebText) ---
dataset_owt = load_dataset("openwebtext", split="train[:5%]")
texts = dataset_owt['text'][:1_500_000]  # sampling (≈300M target tokens)

print(f"Total raw texts loaded: {len(texts):,}")

# --- GPT-3 tokenizer ---
tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-3 BPE tokenizer
print(f"Tokenizer vocab size: {tokenizer.n_vocab:,}")

# Update config vocab_size
config.vocab_size = tokenizer.n_vocab
print(f"Updated config vocab_size to: {config.vocab_size:,}")

# --- Dataset & DataLoader ---
max_seq_len = 256  # adjust based on GPU memory
train_dataset = GPT3Dataset(
    texts, tokenizer, max_seq_len=max_seq_len,  # keep tokenizer's vocab by default
    vocab_size=config.vocab_size, sequential=False  # False → random subsequences
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

# Debug check
sample_in, sample_out = train_dataset[0]
print(f"Sample input shape: {sample_in.shape}, dtype={sample_in.dtype}")
print(f"Sample target shape: {sample_out.shape}, dtype={sample_out.dtype}")
